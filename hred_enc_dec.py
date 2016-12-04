from base import *


# hierarchical seq2seq model
class hred_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size, embedding, learning_rate,
                 decoded=1, mode=0, bn=0):
        self.c_size = c_size
        with tf.variable_scope('hierarchical'):
            self.hred = rnn_cell.GRUCell(self.c_size)
        # batch normalization parameters
        if bn:
            self.scale = tf.Variable(tf.ones([self.decoder_in_size()]), dtype=tf.float32, name='Embedding_b')
            self.offset = tf.Variable(tf.zeros([self.decoder_in_size()]), dtype=tf.float32, name='Embedding_b')
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
                pass

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
                if self.mode < 3:
                    max_len = tf.shape(self.data[i + 1])[1]
                    output, _ = rnn.dynamic_rnn(self.decodernet, self._decoder_input(h, i, max_len),
                                                sequence_length=self.length[i + 1], dtype=tf.float32)
                    dec.reuse_variables()
                    # output: batch_size*max_length*h_size
                    decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
                if self.mode > 1:  # generate response
                    init_word = self.embedded_word(self.start_word(), [self.batch_size, 1, 300])
                    initial_state = tf.zeros([self.batch_size, self.h_size])
                    word_ind, num = 0, 0
                    if self.batch_size == 1:
                        while (word_ind != 2 and num < 20):
                            output, dec_state = rnn.dynamic_rnn(self.decodernet,
                                                                tf.concat(2, [init_word, self._context_input(h, i, 1)]),
                                                                dtype=tf.float32, initial_state=initial_state)
                            dec.reuse_variables()
                            initial_state = dec_state
                            word_ind = tf.argmax(tf.nn.softmax(
                                tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2), 1)
                            init_word = self.embedded_word(tf.one_hot(word_ind, self.vocab_size, dtype=tf.float32),
                                                           [self.batch_size, 1, 300])
                            num += 1
                            decoded_sequence.append(word_ind[0])
            return decoded if self.mode < 2 else decoded_sequence

    @define_scope
    def prediction(self):
        encoder_states = base_enc_dec.encode_word(self)
        h = self.encode_sequence(encoder_states)
        output = self.decode(h, encoder_states)
        return output
