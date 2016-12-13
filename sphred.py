from hred import *


class sphred_enc_dec(hred_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode):
        hred_enc_dec.__init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate,
                              mode)

    """
    sphier-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 0 or 2, update state, else stay unchange
    prev_h: [batch_size*c_size,batch_size*c_size], previous states for two actors
    input: batch_size*h_size
    """

    def hier_level_rnn(self, prev_h, input_vec, mask):
        with tf.variable_scope('hier'):
            state_mask = self.num_seq % 2  # even:0, odd: 1
            prev_state = prev_h[0] * (1 - state_mask) + prev_h[1] * state_mask  # even: prev[0], odd: prev[1]
            _, h_new = self.hiernet(input_vec, prev_state)
            h_masked = h_new * (1 - mask) + prev_state * mask  # update when meeting 0 or 2

            prev_h[0] = h_masked * (1 - state_mask) + prev_h[0] * state_mask  # update when num_seq is even
            prev_h[1] = h_masked * state_mask + prev_h[1] * (1 - state_mask)  # update when num_seq is odd
            self.num_seq += (1 - mask)
            return prev_h

    # return the context input
    # h_s is the hidden state after the hier level
    def context_input(self, h_s):
        state_mask = self.num_seq % 2  # even:0, odd: 1
        return h_s[0] * (1 - state_mask) + h_s[1] * state_mask  # even: h_s[0], odd: h_s[1]

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        self.num_seq = tf.zeros([self.batch_size, 1])  # id of the current speech, increase when meeting 0 or 2
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = [tf.zeros([self.batch_size, self.c_size]), tf.zeros([self.batch_size, self.c_size])]
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, _, h_d = tf.scan(self.run, self.labels, initializer=[init_encode, init_hier, init_decoder])
        return h_d
