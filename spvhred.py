from vhred import *


class spvhred(vhred):
    @init_final
    def __init__(self, labels, length, h_size, c_size, z_size, vocab_size, embedding, batch_size, learning_rate, mode,
                 beam_size=5):
        # prior distribution parameters
        self.scale_cov = 0.1
        self.first_priff = Dense("Latent", z_size, 2 * c_size, nonlinearity=tf.tanh, name='first_priff')
        self.second_priff = Dense("Latent", z_size, z_size, nonlinearity=tf.tanh, name='second_priff')
        self.first_postff = Dense("Latent", z_size, 2 * c_size + h_size, nonlinearity=tf.tanh, name='first_postff')
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
        return 2 * self.c_size + self.z_size

    def hier_level_rnn(self, prev_h, input_vec, mask, eot_mask, num_seq):
        with tf.variable_scope('hier'):
            state_mask = num_seq % 2  # even:0, odd: 1
            prev_state = prev_h[0] * (1 - state_mask) + prev_h[1] * state_mask  # even: prev[0], odd: prev[1]
            _, h_new = self.hiernet(input_vec, prev_state)
            h_masked = h_new * (1 - mask) + prev_state * mask  # update when meeting eou

            prev_h[0] = h_masked * (1 - state_mask) + prev_h[0] * state_mask  # update when num_seq is even
            prev_h[1] = h_masked * state_mask + prev_h[1] * (1 - state_mask)  # update when num_seq is odd
            return prev_h, num_seq + 1 - eot_mask

    """
    prev_h[0]: word-level last state
    prev_h[1]: hier last state
    prev_h[2]: num_seq
    """

    def run_first(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels[0], EOU)
        rolled_mask = self.gen_mask(input_labels[1], EOU)
        eot_mask = self.gen_mask(input_labels[0], EOT)
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        h_s, num_seq = self.hier_level_rnn(prev_h[1], h, mask, eot_mask, prev_h[2])
        return [h, h_s, num_seq]

    # prev_h: kl divergence, decoder state, latent state
    # input_labels: h_s, r_h, labels, num_seq
    def run_second(self, prev_h, input_labels):
        pre_kl, pre_h_d, pre_z = prev_h
        h_s0, h_s1, r_h, label, num_seq = input_labels
        embedding = self.embed_labels(label)
        mask = self.gen_mask(label, EOU) * self.gen_mask(label, EOT)

        # decide how to concat context
        state_mask = num_seq % 2
        h_own = h_s0 * (1 - state_mask) + h_s1 * state_mask  # context of current speaker
        h_other = h_s0 * state_mask + h_s1 * (1 - state_mask)  # context of other speaker
        h_s = tf.concat(1, [h_own, h_other])

        embedding *= mask  # mark first embedding as 0
        # compute posterior
        pri_mean, pri_cov = self.compute_prior(h_s)
        pos_mean, pos_cov = self.compute_post(tf.concat(1, [r_h, h_s]))
        kl = pre_kl + self._kldivergence(pos_mean, pri_mean, pos_cov, pri_cov) * (
            1 - mask)  # divergence increases when meeting eou

        # drop out with 0.25
        drop_mask = tf.cast(tf.random_uniform([self.batch_size, 1], maxval=1) > 0.75, tf.float32)
        unk_embedding = self.embedding_W[1]
        embedding = embedding * (1 - drop_mask) + unk_embedding * drop_mask

        # sample latent variable
        z = self.sampleGaussian(pos_mean, pos_cov)
        z = z * (1 - mask) + pre_z * mask  # update when meeting eou or eot
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(pre_h_d, tf.concat(1, [z, h_s, embedding]), mask)
        return [kl, d, z]

    def scan_step(self):
        num_seq = tf.zeros([self.batch_size, 1])  # id of the current speech, increase when meeting 2
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = [tf.zeros([self.batch_size, self.c_size]), tf.zeros([self.batch_size, self.c_size])]
        init_latent = tf.zeros([self.batch_size, self.z_size])
        kl = tf.zeros([self.batch_size, 1])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        mask = tf.cast(tf.not_equal(self.labels, EOU), tf.float32) * tf.cast(tf.not_equal(self.labels, EOT), tf.float32)
        h, h_s, num_seq = tf.scan(self.run_first, [self.labels, self.rolled_label],
                                  initializer=[init_encode, init_hier, num_seq])
        r_h = tf.reverse(h, dims=[True, False, False])
        r_mask = tf.reverse(mask, [True, False])
        reversed_h = tf.scan(self.reverse_h, [r_h, r_mask], initializer=r_h[0])
        r_h = tf.reverse(reversed_h, dims=[True, False, False])

        kl, h_d, _ = tf.scan(self.run_second, [h_s[0][:-1], h_s[1][:-1], r_h[1:], self.labels[:-1], num_seq[:-1]],
                             initializer=[kl, init_decoder, init_latent])

        return [kl, h_d, h_s, num_seq]

    def decode_bs(self, h_d):
        last_h_s = h_d[2][-1]
        num_seq = h_d[3][-1]
        state_mask = num_seq % 2
        h_s = tf.concat(1, [last_h_s[1], last_h_s[0]]) * state_mask + tf.concat(1, [last_h_s[0], last_h_s[1]]) * (
        1 - state_mask)
        pri_mean, pri_cov = self.compute_prior(h_s)
        z = self.sampleGaussian(pri_mean, pri_cov)
        z_hs = tf.concat(1, [z, h_s])
        prev_d = tf.tanh(tf.matmul(z_hs, self.init_W) + self.init_b)
        inp = tf.zeros([1, 300])
        k = 0
        while k < 15:
            if k == 1:
                z_hs = tf.tile(z_hs, [self.beam_size, 1])   
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                _, d_new = self.decodernet(tf.concat(1, [z_hs, inp]), prev_d)
                prev_d = d_new
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded 

    @exe_once
    def cost(self):
        print('labels:', self.labels[1:])
        y_flat = tf.reshape(self.labels[1:], [-1])  # exclude the first padded label
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction[0], y_flat)
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = tf.reshape(mask * loss, tf.shape(self.labels[1:]))
        # normalized loss per example
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=0) / tf.to_float(self.length)

        # average kl-divergence
        self.num_eos = tf.reduce_sum(tf.cast(tf.equal(self.labels[:-1], EOU), tf.float32)) + tf.reduce_sum(
            tf.cast(tf.equal(self.labels[:-1], EOT), tf.float32))
        avg_kl = self.prediction[1] / self.num_eos
        return tf.reduce_mean(mean_loss_by_example), avg_kl  # average loss of the batch
