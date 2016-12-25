from hred import *
from dense import *


class vhred(hred):
    @init_final
    def __init__(self, labels, length, h_size, c_size, z_size, vocab_size, embedding, batch_size, learning_rate, mode,
                 beam_size=5):
        # prior distribution parameters
        self.scale_cov = 0.1
        self.first_priff = Dense("Latent", z_size, c_size, nonlinearity=tf.tanh, name='first_priff')
        self.second_priff = Dense("Latent", z_size, z_size, nonlinearity=tf.tanh, name='second_priff')
        self.first_postff = Dense("Latent", z_size, c_size + h_size, nonlinearity=tf.tanh, name='first_postff')
        self.second_postff = Dense("Latent", z_size, z_size, nonlinearity=tf.tanh, name='second_postff')
        self.prior_m = Dense("Latent", z_size, z_size, name='prior_mean')
        self.prior_c = Dense("Latent", z_size, z_size, name='prior_cov')
        self.post_m = Dense("Latent", z_size, z_size, name='post_mean')
        self.post_c = Dense("Latent", z_size, z_size, name='post_cov')
        self.z_size = z_size
        hred.__init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode,
                      beam_size)

    @property
    def context_len(self):
        return self.z_size + self.c_size

    """
    generate mean and covariance
    context: batch_size*context_size
    """

    def compute_prior(self, context):
        first_ff = self.first_priff(context)  # batch_size*z_size
        second = self.second_priff(first_ff)
        mean = self.prior_m(second)
        cov = self.prior_c(second)
        return mean, tf.nn.softplus(cov) * self.scale_cov

    def compute_post(self, context):
        first_ff = self.first_postff(context)  # batch_size*z_size
        second = self.second_postff(first_ff)
        mean = self.post_m(second)
        cov = self.post_c(second)
        return mean, tf.nn.softplus(cov) * self.scale_cov

    def sampleGaussian(self, mu, sigma):
        """(Differentiable) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(sigma), name="epsilon")
            print('mu:', mu)
            return mu + epsilon * sigma  # N(mu, I * sigma**2)

    # KL-divergence within one batch
    def __kldivergence(self, mu1, mu2, s1, s2):
        """Average (Gaussian) Kullback-Leibler divergence KL(g1||g2), per training batch"""
        # (tf.Tensor, tf.Tensor) -> tf.Tensor
        with tf.name_scope("KL_divergence"):
            kl = 0.5 * (tf.reduce_sum(tf.log(tf.abs(s2)) - tf.log(tf.abs(s1)) + s1 / s2 + (
                mu2 - mu1) ** 2 / s2, reduction_indices=[1]) - self.z_size)
            return tf.reshape(kl, [self.batch_size, 1])

    def run_first(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels[0], EOU)
        rolled_mask = self.gen_mask(input_labels[1], EOU)
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        h_s = self.hier_level_rnn(prev_h[1], h, mask)
        return [h, h_s]

    # prev_h: kl divergence, decoder state, latent state
    # input_labels: h_s, r_h, labels
    def run_second(self, prev_h, input_labels):
        pre_kl, pre_h_d, pre_z = prev_h
        h_s, r_h, label = input_labels
        embedding = self.embed_labels(label)
        mask = self.gen_mask(label, EOU)
        embedding *= mask  # mark first embedding as 0

        print('h_s:', h_s)
        print('r_h:', r_h)
        # compute posterior
        pri_mean, pri_cov = self.compute_prior(h_s)
        pos_mean, pos_cov = self.compute_post(tf.concat(1, [r_h, h_s]))
        kl = pre_kl + self.__kldivergence(pos_mean, pri_mean, pos_cov, pri_cov) * (
            1 - mask)  # divergence increases when meeting eou

        # drop out with 0.25
        drop_mask = tf.cast(tf.random_uniform([self.batch_size, 1], maxval=1) > 0.75, tf.float32)
        unk_embedding = self.embedding_W[1]
        embedding = embedding * (1 - drop_mask) + unk_embedding * drop_mask

        # sample latent variable
        z = self.sampleGaussian(pos_mean, pos_cov)
        z = z * (1 - mask) + pre_z * mask  # update when meeting eou
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(pre_h_d, tf.concat(1, [z, h_s, embedding]), mask)
        return [kl, d, z]

    # prev_h: reversed h[i]
    # input_labels: reversed h[i], mask
    def reverse_h(self, prev_h, input_labels):
        h, mask = input_labels
        mask = tf.reshape(mask, [self.batch_size, 1])
        new_h = prev_h * mask + h * (1 - mask)
        return new_h

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = tf.zeros([self.batch_size, self.c_size])
        init_latent = tf.zeros([self.batch_size, self.z_size])
        kl = tf.zeros([self.batch_size,1])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        mask = tf.cast(tf.not_equal(self.labels, EOU), tf.float32)
        h, h_s = tf.scan(self.run_first, [self.labels, self.rolled_label], initializer=[init_encode, init_hier])
        r_h = tf.reverse(h, dims=[True, False, False])
        r_mask = tf.reverse(mask, [True, False])
        print('r_h:', r_h)
        print('mask:', mask)
        print('r_h[0]', r_h[0])
        reversed_h = tf.scan(self.reverse_h, [r_h, r_mask], initializer=r_h[0])
        print('reversed_h:', reversed_h)
        print('h_s:', h_s)
        r_h = tf.reverse(reversed_h, dims=[True, False, False])
        kl, h_d, _ = tf.scan(self.run_second, [h_s[:-1], r_h[1:], self.labels[:-1]],
                             initializer=[kl, init_decoder, init_latent])

        return [kl, h_d]

    # return output layer
    @exe_once
    def prediction(self):
        h_d = self.scan_step()
        if self.mode == 2:
            sequences = self.decode_bs(h_d)
            return sequences
        predicted = tf.reshape(h_d[1], [-1, self.h_size])  # exclude the last prediction
        output = tf.matmul(predicted, self.output_W) + self.output_b  # (max_len*batch_size)*vocab_size
        return output, tf.reduce_sum(h_d[0][-1])  # kldivergence

    @exe_once
    def cost(self):
        y_flat = tf.reshape(self.labels[1:], [-1])  # exclude the first padded label
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction[0], y_flat)
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = tf.reshape(mask * loss, tf.shape(self.labels[1:]))
        # normalized loss per example
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=0) / tf.to_float(self.length)

        # average kl-divergence
        self.num_eos = tf.reduce_sum(tf.cast(tf.equal(self.labels[:-1], EOU), tf.float32))
        avg_kl = self.prediction[1] / self.num_eos
        return tf.reduce_mean(mean_loss_by_example), avg_kl  # average loss of the batch

    @exe_once
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost[0] + self.cost[1] * tf.to_float(tf.reduce_min([1, global_step / 75000])),
                                  global_step=global_step)
        return global_step, train_op, tf.to_float(tf.reduce_min([1, global_step / 75000]))
