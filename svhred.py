from vhred import *


class svhred(vhred):
    # e_size: size of label embedding
    # obj: max_len*batch_size, objective labels
    @init_final
    def __init__(self, labels, obj, length, h_size, c_size, z_size, e_size, vocab_size, embedding, batch_size,
                 learning_rate, mode,
                 beam_size=5):
        # prior distribution parameters
        self.scale_cov = 0.1
        self.first_priff = Dense("Latent", z_size, c_size + e_size, nonlinearity=tf.tanh, name='first_priff')
        self.second_priff = Dense("Latent", z_size, z_size, nonlinearity=tf.tanh, name='second_priff')
        self.first_postff = Dense("Latent", z_size, c_size + h_size + e_size, nonlinearity=tf.tanh,
                                  name='first_postff')
        self.second_postff = Dense("Latent", z_size, z_size, nonlinearity=tf.tanh, name='second_postff')
        self.prior_m = Dense("Latent", z_size, z_size, name='prior_mean')
        self.prior_c = Dense("Latent", z_size, z_size, name='prior_cov')
        self.post_m = Dense("Latent", z_size, z_size, name='post_mean')
        self.post_c = Dense("Latent", z_size, z_size, name='post_cov')

        with tf.variable_scope('Latent'):
            self.obj_W = tf.get_variable('Obj_W', initializer=tf.random_normal([2, e_size]))
            self.obj_b = tf.get_variable('Obj_b', initializer=tf.zeros([e_size]))
        # classify parameters
        self.pred = Dense("Latent", 2, z_size, name='Classify')

        self.obj = obj
        self.e_size = e_size
        self.z_size = z_size
        hred.__init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode,
                      beam_size)

    @property
    def context_len(self):
        return self.c_size + self.z_size + self.e_size

    # prev_h: kl divergence, decoder state, latent state
    # input_labels: h_s, r_h, labels, num_seq
    def run_second(self, prev_h, input_labels):
        pre_kl, pre_err, pre_h_d, pre_z = prev_h
        h_s, r_h, r_obj, label = input_labels
        embedding = self.embed_labels(label)
        obj_embedding = tf.gather(self.obj_W, r_obj) + self.obj_b  # obj embedding, batch_size*e_size
        mask = self.gen_mask(label, EOU) * self.gen_mask(label, EOT)

        embedding *= mask  # mark first embedding as 0
        # compute posterior
        pri_mean, pri_cov = self.compute_prior(tf.concat(1, [obj_embedding, h_s]))
        pos_mean, pos_cov = self.compute_post(tf.concat(1, [r_h, obj_embedding, h_s]))
        kl = pre_kl + self._kldivergence(pos_mean, pri_mean, pos_cov, pri_cov) * (
            1 - mask)  # divergence increases when meeting eou or eot

        # drop out with 0.25
        drop_mask = tf.cast(tf.random_uniform([self.batch_size, 1], maxval=1) > 0.75, tf.float32)
        unk_embedding = self.embedding_W[1]
        embedding = embedding * (1 - drop_mask) + unk_embedding * drop_mask

        # sample latent variable
        z = self.sampleGaussian(pos_mean, pos_cov)
        z = z * (1 - mask) + pre_z * mask  # update when meeting eou or eot

        pred_err = pre_err + tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred(z), r_obj) * (1 - mask)
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(pre_h_d, tf.concat(1, [z, h_s, embedding]), mask)
        return [kl, pred_err, d, z]

    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = tf.zeros([self.batch_size, self.c_size])
        init_latent = tf.zeros([self.batch_size, self.z_size])
        kl = tf.zeros([self.batch_size, 1])

        pred_err = tf.zeros([self.batch_size, 1])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        mask = tf.cast(tf.not_equal(self.labels, EOU), tf.float32) * tf.cast(tf.not_equal(self.labels, EOT), tf.float32)
        h, h_s = tf.scan(self.run_first, [self.labels, self.rolled_label],
                         initializer=[init_encode, init_hier])
        r_h = tf.reverse(h, dims=[True, False, False])

        r_mask = tf.reverse(mask, [True, False])
        reversed_h = tf.scan(self.reverse_h, [r_h, r_mask], initializer=r_h[0])
        r_h = tf.reverse(reversed_h, dims=[True, False, False])

        kl, pred_err, h_d, _ = tf.scan(self.run_second,
                                       [h_s[:-1], r_h[1:], self.obj, self.labels[:-1]],
                                       initializer=[kl, pred_err, init_decoder, init_latent])

        return [kl, pred_err, h_d]

    # return output layer
    @exe_once
    def prediction(self):
        h_d = self.scan_step()
        if self.mode == 2:
            sequences = self.decode_bs(h_d)
            return sequences
        predicted = tf.reshape(h_d[2], [-1, self.h_size])  # exclude the last prediction
        output = self.output1(predicted)  # (max_len*batch_size)*vocab_size
        output = self.output2(output)
        return output, tf.reduce_sum(h_d[0][-1]), tf.reduce_sum(h_d[1][-1])  # kldivergence and pred_err

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
        avg_err = self.prediction[2] / self.num_eos
        return tf.reduce_mean(mean_loss_by_example), avg_kl, avg_err  # average loss of the batch

    @exe_once
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(
            self.cost[0] + self.cost[2] + self.cost[1] * tf.to_float(tf.reduce_min([1, global_step / 75000])),
            global_step=global_step)
        return global_step, train_op, tf.to_float(tf.reduce_min([1, global_step / 75000]))
