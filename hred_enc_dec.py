from base import *


# hierarchical seq2seq model
class hred_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size, embedding,learning_rate,
                 decoded=1,mode=0):
        base_enc_dec.__init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding,learning_rate,
                              decoded,mode)
        self.c_size = c_size
        with tf.variable_scope('hierarchical'):
            self.hred = rnn_cell.GRUCell(self.c_size)

    # input of context for decoding sequences
    def _context_input(self, h, i, max_len):
        context_input = tf.tile(h[i + 1], [max_len, 1])
        return tf.reshape(context_input,
                          [tf.shape(h[i + 1])[0], max_len, self.c_size])  # batch_size*max_len*c_size

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
        decoded = []
        with tf.variable_scope('decode') as dec:
            # mapping to vocab probability
            W2 = tf.Variable(tf.zeros([self.decodernet.output_size, self.vocab_size]), dtype=tf.float32,
                             name='Output_W')
            b2 = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='Output_b')
            # decode, starts from the context state before the first decoded sequence
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                max_len = tf.shape(self.data[i + 1])[1]
                output, _ = rnn.dynamic_rnn(self.decodernet, tf.concat(2, [self._sentence_input(i, max_len),
                                                                           self._context_input(h, i, max_len)]),
                                            sequence_length=self.length[i + 1], dtype=tf.float32)
                dec.reuse_variables()
                # output: batch_size*max_length*h_size
                decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), W2) + b2)

            return decoded

    @define_scope
    def prediction(self):
        encoder_states = base_enc_dec.encode_word(self)
        h = self.encode_sequence(encoder_states)
        output = self.decode(h, encoder_states)
        return output
