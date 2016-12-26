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
    # input_labels: h_s, r_h, labels， num_seq
    def run_second(self, prev_h, input_labels):
        pre_kl, pre_h_d, pre_z = prev_h
        h_s0, h_s1, r_h, label, num_seq = input_labels
        embedding = self.embed_labels(label)
        mask = self.gen_mask(label, EOU)
        eot_mask = self.gen_mask(label, EOT)

        # decide how to concat context
        state_mask = num_seq % 2
        h_own = h_s0 * (1 - state_mask) + h_s1 * state_mask  # context of current speaker
        h_other = h_s0 * state_mask + h_s1 * (1 - state_mask)  # context of other speaker
        h_s = tf.concat(1, [h_own, h_other])

        embedding *= (mask * eot_mask)  # mark first embedding as 0
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
        z = z * (1 - mask) + pre_z * mask  # update when meeting eou
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
        mask = tf.cast(tf.not_equal(self.labels, EOU), tf.float32)
        h, h_s, num_seq = tf.scan(self.run_first, [self.labels, self.rolled_label],
                                  initializer=[init_encode, init_hier, num_seq])
        r_h = tf.reverse(h, dims=[True, False, False])
        r_mask = tf.reverse(mask, [True, False])
        reversed_h = tf.scan(self.reverse_h, [r_h, r_mask], initializer=r_h[0])
        r_h = tf.reverse(reversed_h, dims=[True, False, False])

        print('h_s:', h_s)
        print('num_seq:', num_seq)
        print('r_h:', r_h)
        print('self.labels:', self.labels[:-1])
        kl, h_d, _ = tf.scan(self.run_second, [h_s[0][:-1], h_s[1][:-1], r_h[1:], self.labels[:-1], num_seq[:-1]],
                             initializer=[kl, init_decoder, init_latent])

        return [kl, h_d]
