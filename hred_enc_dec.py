from base import *


# hierarchical seq2seq model
class hred_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size, embedding,
                 learning_rate,
                 decoded=1, mode=0, bn=0, model=0, beam_size=5):
        # model: 0-hierarchical GRU, 1-hred, 2-sphred
        length = tf.cast(length, tf.int32)
        self.c_size = c_size
        self.beam_size = beam_size
        self.model = model
        self.log_beam_probs, self.beam_path, self.output_beam_symbols, self.beam_symbols = [], [], [], []
        with tf.variable_scope('encode'):
            self.encodernet = rnn_cell.GRUCell(h_size)
        with tf.variable_scope('hierarchical'):
            self.hred = rnn_cell.GRUCell(c_size)
        # batch normalization parameters
        if bn:
            self.scale = tf.Variable(tf.ones([self.decoder_in_size()]), dtype=tf.float32, name='Bn_scale')
            self.offset = tf.Variable(tf.zeros([self.decoder_in_size()]), dtype=tf.float32, name='Bn_offset')
        base_enc_dec.__init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding,
                              learning_rate,
                              decoded, mode, bn)

    # input size to the final decoder
    def decoder_in_size(self):
        return self.c_size + 300

    # input of context for decoding sequences
    def _context_input(self, h, i, max_len):
        context_input = tf.tile(h[i + 1], [max_len, 1])
        return tf.reshape(context_input,
                          [tf.shape(h[i + 1])[0], max_len, self.c_size])  # batch_size*max_len*c_size

    def _decoder_input(self, h, i, max_len):
        decoder_input = tf.concat(2, [self._sentence_input(i, max_len), self._context_input(h, i, max_len)])
        # batch normalisation
        if self.bn:
            # training mode, use batch mean and variance
            if self.mode == 0:
                mean, variance = tf.nn.moments(decoder_input, [0, 1])
                return tf.nn.batch_normalization(decoder_input, mean, variance, self.offset, self.scale, 0.001)
            # testing mode, use expexted mean and variance
            else:
                return tf.nn.batch_normalization(decoder_input, self.mean, self.variance, self.offset, self.scale,
                                                 0.001)
        return decoder_input

    def decode_every_word(self, hier, dummy):
        decoded = []
        with tf.variable_scope('decode') as dec:
            # probability for the first sequence
            hier_0 = tf.slice(tf.pack(hier[0]), [0, 0, 0],
                              [tf.shape(self.data[0])[1] - 1, self.batch_size, self.c_size])
            hier_0 = tf.reshape(hier_0, [-1, self.batch_size, self.c_size])
            seq_input = tf.transpose(hier_0, perm=[1, 0, 2])
            if self.model == 1:
                dummy_0 = tf.slice(tf.pack(dummy[0]), [0, 0, 0],
                                   [tf.shape(self.data[0])[1] - 1, self.batch_size, self.h_size])
                dummy_0 = tf.reshape(dummy_0, [-1, self.batch_size, self.h_size])
                seq_input = tf.concat(2, [tf.transpose(hier_0, perm=[1, 0, 2]), tf.transpose(dummy_0, perm=[1, 0, 2])])
            output, _ = rnn.dynamic_rnn(self.decodernet, seq_input, dtype=tf.float32)
            # the probability for the first word is zero padded, not trainable
            first_prob = tf.concat(0, [tf.zeros([self.batch_size, self.vocab_size]),
                                       tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]),
                                                 self.W2) + self.b2])
            decoded.append(first_prob)
            dec.reuse_variables()
            # probability for the next sequences
            for i in range(1, self.num_seq):

                hier_i = tf.slice(tf.pack(hier[i]), [0, 0, 0],
                                  [tf.shape(self.data[i])[1] - 1, self.batch_size, self.c_size])
                whole_last_hier = tf.pack(hier[i - 1])  # 100*batch_size*h_size
                last_hier = []
                for j in range(self.batch_size):
                    last_hier.append(
                        tf.reshape(tf.slice(whole_last_hier, [self.length[i - 1, j] - 1, j, 0], [1, 1, self.c_size]),
                                   [self.c_size]))
                last_hier = tf.pack(last_hier)  # batch_size*h_size
                hier_i = tf.concat(0, [tf.reshape(last_hier, [1, self.batch_size, self.c_size]), hier_i])
                seq_input = tf.transpose(hier_i, perm=[1, 0, 2])
                if self.model == 1:
                    dummy_i = tf.slice(tf.pack(dummy[i]), [0, 0, 0],
                                       [tf.shape(self.data[i])[1] - 1, self.batch_size, self.h_size])
                    last_dummy = dummy[i - 1][-1]

                    dummy_i = tf.concat(0, [tf.reshape(last_dummy, [1, self.batch_size, self.h_size]), dummy_i])
                    seq_input = tf.concat(2, [tf.transpose(hier_i, perm=[1, 0, 2]),
                                              tf.transpose(dummy_i, perm=[1, 0, 2])])
                output, _ = rnn.dynamic_rnn(self.decodernet, seq_input, dtype=tf.float32)
                decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
        return decoded

    # new encode in word-level, for every sequence, a 0-word is appended to the beginning
    # hier[i]: the hierarchical state after the i-th word, shape=batch_size*h_size
    # dummy[i]: the sum of word-level state until after the i-th word, shape=batch_size*c_size
    def encode_every_word(self):
        hier, dummy = [], []

        # encode in word-level
        def rnnstep(j, seq_input, init_s):
            with tf.variable_scope('rnn'):
                _, h_j = rnn.rnn(self.encodernet, [
                    tf.reshape(tf.slice(seq_input, [0, j, 0], [self.batch_size, 1, 300]), [self.batch_size, 300])],
                                 dtype=tf.float32,
                                 initial_state=init_s)
            return h_j

        # encode in hierarchical level
        def hrnnstep(h_j, init_s):
            with tf.variable_scope('hrnn'):
                _, h_sj = rnn.rnn(self.hred, [h_j], dtype=tf.float32,
                                  initial_state=init_s)
            return h_sj

        with tf.variable_scope('encode') as enc:
            for i in range(self.num_seq):
                hier.append([tf.Variable(tf.zeros([self.batch_size, self.c_size]))] * 100)
                dummy.append([tf.Variable(tf.zeros([self.batch_size, self.h_size]))] * 100)
                initial_state = tf.zeros([self.batch_size, self.h_size])
                h_init = tf.zeros([self.batch_size, self.c_size])
                concatenated = tf.reshape(self.data[i], [-1, self.vocab_size])
                embedded = tf.matmul(concatenated, self.W) + self.b
                seq_input = tf.reshape(embedded,
                                       [self.batch_size, tf.shape(self.data[i])[1], 300])  # batch_size*(max_len)*300
                for j in range(100):
                    # if j<max_len, return hidden state, else, zeros state
                    print(j)
                    h_j = tf.cond(tf.less(j, tf.shape(self.data[i])[1]), lambda: rnnstep(j, seq_input, initial_state),
                                  lambda: tf.zeros([self.batch_size, self.h_size]))
                    initial_state = h_j
                    if self.model > 0:
                        for k in range(j, 100):
                            tf.assign_add(dummy[i][j], h_j)
                    hs_j = tf.cond(tf.less(j, tf.shape(self.data[i])[1]), lambda: hrnnstep(h_j, h_init),
                                   lambda: tf.zeros([self.batch_size, self.c_size]))
                    tf.assign(hier[i][j], hs_j)
                    h_init = hs_j
                    enc.reuse_variables()
        return hier, dummy

    # sequence-level encoding, keep needed states for {decoded} last sequences
    # e is word-level hidden states
    def encode_sequence(self, e):
        # h_state[i] keeps context state for the (i+1)th sequence, with an additional initializing context
        # when no need for decoding, also assigned as zeros
        h_state = [tf.zeros([self.batch_size, self.c_size])] * (self.num_seq - self.decoded)
        # encode in sentence-level
        with tf.variable_scope('hierarchical') as hier:
            initial_state = tf.zeros([self.batch_size, self.c_size])
            # run sequence-level rnn until decoding is needed, these states don't need to be stored
            if self.num_seq - self.decoded > 1:
                _, initial_state = rnn.dynamic_rnn(self.hred,
                                                   tf.transpose(e[1:self.num_seq - self.decoded], perm=[1, 0, 2]),
                                                   dtype=tf.float32, initial_state=initial_state)
                hier.reuse_variables()
            # run senquence-level rnn while keeping all the hidden states
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                if i > 0:
                    hier.reuse_variables()
                # every time only moves one step towards, the intermediate states are stored
                _, initial_state = rnn.dynamic_rnn(self.hred, tf.reshape(e[i + 1], [self.batch_size, 1, self.h_size]),
                                                   dtype=tf.float32, initial_state=initial_state)
                h_state.append(initial_state)
        return h_state

    # decode specified {decoded} senquences
    # h and e are sequence-level and word-level hidden states
    # return a list in which each item is of size batch_size*max_length*vocab_size
    def decode(self, h, e):
        # decoded array keeps the output for all decoded sequences
        decoded, decoded_sequence = [], []
        with tf.variable_scope('decode') as dec:
            # decode, starts from the context state before the first decoded sequence
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                if self.mode < 2:
                    max_len = tf.shape(self.data[i + 1])[1]
                    # mean_context=tf.reduce_mean(tf.abs(self._context_input(h, 1, max_len)))
                    # mean_word=tf.reduce_mean(tf.abs(self._sentence_input(1,max_len)))
                    # return self._decoder_input(h, i, max_len)
                    output, _ = rnn.dynamic_rnn(self.decodernet, self._decoder_input(h, i, max_len),
                                                sequence_length=self.length[i + 1], dtype=tf.float32)
                    dec.reuse_variables()
                    # output: batch_size*max_length*h_size
                    decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
                if self.mode == 1:  # generate response
                    k = 0
                    inp = self.embedded_word(self.start_word(), [1, 1, 300])
                    initial_state = tf.zeros([1, self.h_size])
                    state = initial_state
                    outputs = []
                    prev = None
                    state_size = int(initial_state.get_shape().with_rank(2)[1])
                    while k < 12:
                        if k == 1:
                            h[i + 1] = tf.tile(h[i + 1], [self.beam_size, 1])
                        if prev is not None:
                            inp = self.beam_search(prev, k)
                            shape = inp.get_shape()
                            inp = tf.reshape(inp, [int(shape[0]), 1, int(shape[1])])
                        if k > 0:
                            dec.reuse_variables()
                        length = 1 if k == 0 else self.beam_size
                        output, state = rnn.dynamic_rnn(self.decodernet,
                                                        tf.concat(2, [inp, self._context_input(h, i, 1)]),
                                                        dtype=tf.float32, initial_state=state)

                        prev = output
                        if k == 0:
                            state = tf.tile(state, [self.beam_size, 1])
                        k += 1
                    decoded_sequence = tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded if self.mode < 1 else decoded_sequence

    def beam_search(self, prev, k):
        probs = tf.log(tf.nn.softmax(
            tf.matmul(tf.reshape(prev, [-1, self.decodernet.output_size]), self.W2) + self.b2))
        minus_probs = [0.0 for i in range(self.vocab_size)]
        # minus_probs[1] = -1e20
        minus_probs[2] = -1e20
        # minus_probs[10963] = -1e20
        # minus_probs[428] = -1e20
        probs = probs + tf.constant(minus_probs)
        if k > 1:
            probs = tf.reshape(probs + self.log_beam_probs[-1],
                               [-1, self.beam_size * self.vocab_size])

        best_probs, indices = tf.nn.top_k(probs, self.beam_size)
        indices = tf.squeeze(tf.reshape(indices, [-1, 1]))
        best_probs = tf.reshape(best_probs, [-1, 1])

        symbols = indices % self.vocab_size  # Which word in vocabulary.
        beam_parent = indices // self.vocab_size  # Which hypothesis it came from.

        self.beam_path.append(beam_parent)
        symbols_live = symbols
        if k > 1:
            symbols_history = tf.gather(self.output_beam_symbols[-1], beam_parent)
            symbols_live = tf.concat(1, [tf.reshape(symbols_history, [-1, k - 1]), tf.reshape(symbols, [-1, 1])])
        self.output_beam_symbols.append(symbols_live)
        self.beam_symbols.append(symbols)
        self.log_beam_probs.append(best_probs)
        concatenated = tf.reshape(
            tf.one_hot(symbols, depth=self.vocab_size, dtype=tf.float32), [-1, self.vocab_size])
        embedded = tf.matmul(concatenated, self.W) + self.b
        emb_prev = tf.reshape(embedded, [self.beam_size, 300])
        return emb_prev

    @define_scope
    def prediction(self):
        hier, dummy = self.encode_every_word()
        de_outputs = self.decode_every_word(hier, dummy)
        return de_outputs
