from base import *


class hred_enc_dec(base_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode):
        self.c_size = c_size
        with tf.variable_scope('hier'):
            self.hiernet = rnn_cell.GRUCell(c_size)
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode)

    """
    hier-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 0 or 2, update state, else stay unchange
    prev_h: batch_size*c_size
    input: batch_size*h_size
    """

    def hier_level_rnn(self, prev_h, input_vec, mask):
        with tf.variable_scope('hier'):
            _, h_new = self.hiernet(input_vec, prev_h)
            h_masked = h_new * (1 - mask) + prev_h * mask  # update when meeting 0 or 2
            return h_masked

    """
    prev_h[0]: word-level last state
    prev_h[1]: hier last state
    prev_h[2]: decoder last state
    hier encoder-decoder model
    """

    def run(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels[0])
        rolled_mask = self.gen_mask(input_labels[1])
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        h_s = self.hier_level_rnn(prev_h[1], h, mask)
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(prev_h[2], tf.concat(1, [h_s, embedding]), rolled_mask)
        return [h, h_s, d]

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = tf.zeros([self.batch_size, self.c_size])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, _, h_d = tf.scan(self.run, [self.labels, self.rolled_label],
                            initializer=[init_encode, init_hier, init_decoder])
        return h_d
