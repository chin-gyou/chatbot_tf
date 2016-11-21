from sphred_enc_dec import *
from dense import *

"""
mu: [batch_size*z_size]
log_sigma: [batch_size*z_size]
emotions: [num_seq*batch_size], every emotion signal is a scalar
emotion_size: number of class of emotion scalar
z_size: size of latent vector
"""


class em_sp_enc_dec(sphred_enc_dec):
    @init_final
    def __init__(self, data, labels, length, emotions, emotion_size, h_size, e_size, c_size, z_size, batch_size,
                 num_seq, vocab_size,
                 embedding, learning_rate,decoded=1,mode=0):
        sphred_enc_dec.__init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size,
                                embedding, learning_rate,decoded,mode)
        self.z_size = z_size
        self.emotions = emotions
        self.emotion_size = emotion_size

        # map emotion to embedding parameters, embedding is of the same size as c_size
        self.emo = Dense("emotion_to_embedding", self.c_size)

        # embedding to predict y label parameters
        self.predict_emo = Dense('predict emotion', self.emotion_size)

        # prior distribution parameters
        self.pri_u = Dense("prior_u", self.z_size)
        self.pri_ls = Dense("prior_logsigma", self.z_size)

        # posterior distritbution parameters
        self.pos_u = Dense("posterior_u", self.z_size)
        self.pos_ls = Dense("posterior_logsigma", self.z_size)

    # input of context for decoding sequences
    # return context_input(batch_size*max_len*3*c_size), kldivergence between prior and posterior, cross entropy of posterior prediction
    def _context_input(self, h, i, max_len):

        # context to generate prior and posterior distribution
        context_pri = tf.concat(concat_dim=1, values=[h[i], h[i + 1]])
        context_pos = tf.concat(concat_dim=1, values=[h[i + 1], h[i + 2]])
        # generate latent z from the concatenated vector of context and emo_input

        z, pri_mu, pri_log_sigma = self.generate_z(context_pri, self.emotions[i+1])
        pos_mu, pos_log_sigma = self.compute_dist(context_pos, self.emotions[i+1])
        context = tf.concat(concat_dim=1, values=[context_pri, z])
        # probability of label
        prob = self.predict_emo(context_pri)
        pred_error = tf.nn.sparse_softmax_cross_entropy_with_logits(prob, self.emotions[i + 1])
        # replicate context max_len times
        context_input = tf.tile(context, [max_len, 1])
        return tf.reshape(context_input,
                          [tf.shape(context)[0], max_len, 2 * self.c_size + self.z_size]), self.__kldivergence(pri_mu,
                                                                                                               pri_log_sigma,
                                                                                                               pos_mu,
                                                                                                               pos_log_sigma), pred_error

    # compute the gaussian distribution of latent variable z
    # contexts: batch_size*context_size
    # emotion: [batch_size], scalar
    # return mu and logsigma
    def compute_dist(self, contexts, emotion, option="prior"):
        # transfer scalar emotion to one-hot vector then mapping to c_size-vector
        emo_input = tf.one_hot(emotion, depth=self.emotion_size, dtype=tf.float32)
        emotion_embed = self.emo(emo_input)
        # input to generate distribution. batch_size*3*c_size
        dist_input = tf.concat(concat_dim=1, values=[contexts, emotion_embed])
        # genetate distribution
        if option == 'prior':
            return self.pri_u(dist_input), self.pri_ls(dist_input)
        if option == 'posterior':
            return self.pos_u(dist_input), self.pos_ls(dist_input)

    # context: batch_size*context_size
    # emotion: batch_size*emotion_size
    # return generated latent variable z and corresponding parameters
    def generate_z(self, contexts, emotion):
        mu, log_sigma = self.compute_dist(contexts, emotion)
        return self.sampleGaussian(mu, log_sigma), mu, log_sigma

    # KL-divergence within one batch
    def __kldivergence(self, mu1, mu2, log_sigma1, log_sigma2):
        """Average (Gaussian) Kullback-Leibler divergence KL(g1||g2), per training batch"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            s1, s2 = tf.exp(log_sigma1), tf.exp(log_sigma2)
            return 0.5 * (tf.reduce_sum(log_sigma2 - log_sigma1 + s1 / s2) / self.batch_size - self.z_size + (
                mu2 - mu1) ** 2 / s2 / self.batch_size)

    def sampleGaussian(self, mu, log_sigma):
        """(Differentiable) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)

    # decode specified {decoded} senquences
    # h and e are sequence-level and word-level hidden states
    def decode(self, h, e):
        # decoded array keeps the output for all decoded sequences
        decoded, kldiv, pred_errors = [], [], []
        with tf.variable_scope('decode') as dec:
            # decode, starts from the context state before the first decoded sequence
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                max_len = tf.shape(self.data[i + 1])[1]
                context_input, kldivergence, pred_error = self._context_input(h, i, max_len)
                output, _ = rnn.dynamic_rnn(self.decodernet,
                                            tf.concat(2, [self._sentence_input(i, max_len), context_input]),
                                            sequence_length=self.length[i + 1], dtype=tf.float32)
                dec.reuse_variables()
                decoded.append(output)
                kldiv.append(kldivergence)
                pred_errors.append(pred_error)
            W2 = tf.Variable(tf.zeros([self.decodernet.output_size, self.vocab_size]), dtype=tf.float32,
                             name='Output_W')
            b2 = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='Output_b')

            logits_flat = tf.matmul(tf.reshape(decoded, [-1, self.decodernet.output_size]), W2) + b2
            return logits_flat, kldiv, pred_errors

    @define_scope
    def prediction(self):
        encoder_states = base_enc_dec.encode_word(self)
        h = self.encode_sequence(encoder_states)
        output, kldiv, pred_error = self.decode(h, encoder_states)
        return output, kldiv, pred_error

    # cost computes the loss when decoding the specified {decoded} sequences
    # returned loss is the mean loss for every decoded sequence
    @define_scope
    def cost(self):
        y_flat = tf.reshape(self.labels[-self.decoded:], [-1])
        output, kldiv, pred_error = self.prediction
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(output, y_flat)
        mean_cross_entropy = self.mean_cross_entropy(y_flat, losses)
        kldiv_loss = tf.reduce_mean(kldiv)
        pred_error = tf.reduce_mean(pred_error)
        return mean_cross_entropy + kldiv_loss + pred_error
