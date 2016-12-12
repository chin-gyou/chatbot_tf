from base import *


# hierarchical seq2seq model
class hred_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size, embedding, learning_rate,
                 decoded=1, mode=0, bn=0, beam_size=5, bi=0):
        self.c_size = c_size
        self.beam_size = beam_size
        self.log_beam_probs, self.beam_path,self.output_beam_symbols, self.beam_symbols = [], [], [],[]
        with tf.variable_scope('hierarchical'):
            self.hred = rnn_cell.GRUCell(self.c_size)
        # batch normalization parameters
        if bn:
            self.scale = tf.Variable(tf.ones([self.decoder_in_size()]), dtype=tf.float32, name='Bn_scale')
            self.offset = tf.Variable(tf.zeros([self.decoder_in_size()]), dtype=tf.float32, name='Bn_offset')
        base_enc_dec.__init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding,
                              learning_rate,
                              decoded, mode, bn, 0, bi)

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
                pass
        return decoder_input

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
                if i > (self.num_seq - self.decoded - 1):
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
                    output, _ = rnn.dynamic_rnn(self.decodernet, self._decoder_input(h, i, max_len),
                                                sequence_length=self.length[i + 1], dtype=tf.float32)
                    dec.reuse_variables()
                    # output: batch_size*max_length*h_size
                    decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
		if self.mode == 2:  # generate response
                    k = 0
                    inp = self.embedded_word(self.start_word(), [1, 1, 300])
                    initial_state = tf.zeros([1, self.h_size])
                    state = initial_state
                    outputs = []
                    prev = None
                    state_size = int(initial_state.get_shape().with_rank(2)[1])
                    while k < 12:
			if k == 1:
                            h[i+1] = tf.tile(h[i+1], [self.beam_size, 1])
                        if prev is not None:
                            inp = self.beam_search(prev, k)
                            shape = inp.get_shape()
                            inp = tf.reshape(inp, [int(shape[0]),1, int(shape[1])])
                        if k > 0:
                            dec.reuse_variables()
                        length = 1 if k == 0 else self.beam_size
                        output, state = rnn.dynamic_rnn(self.decodernet,
                                                        tf.concat(2, [inp, self._context_input(h, i, 1)]),
                                                        dtype=tf.float32, initial_state=state)
			
                        prev = output
                        if k == 0:
                            state = tf.tile(state,[self.beam_size,1])
                        k += 1
                    decoded_sequence =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded if self.mode < 2 else decoded_sequence

    def beam_search(self, prev, k):
        probs = tf.log(tf.nn.softmax(
                    tf.matmul(tf.reshape(prev, [-1, self.decodernet.output_size]), self.W2) + self.b2))
        minus_probs = [0 for i in range(self.vocab_size)]
        minus_probs[1] = -1e20
        probs = probs +  tf.constant(minus_probs)
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
            symbols_live = tf.concat(1,[tf.reshape(symbols_history,[-1,k-1]), tf.reshape(symbols, [-1, 1])])
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
        encoder_states = base_enc_dec.encode_word(self)
        h = self.encode_sequence(encoder_states)
        output = self.decode(h, encoder_states)
        return output
