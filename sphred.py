from hred import *

class sphred(hred):
    @init_final
    def __init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode):
        hred.__init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate,
                              mode)

    @property
    def context_len(self):
        return 2 * self.c_size

    """
    sphier-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 0 or 2, update state, else stay unchange
    prev_h: [batch_size*c_size,batch_size*c_size], previous states for two actors
    input: batch_size*h_size
    """

    def hier_level_rnn(self, prev_h, input_vec, mask, num_seq):
        with tf.variable_scope('hier'):
            state_mask = num_seq % 2  # even:0, odd: 1
            prev_state = prev_h[0] * (1 - state_mask) + prev_h[1] * state_mask  # even: prev[0], odd: prev[1]
            _, h_new = self.hiernet(input_vec, prev_state)
            h_masked = h_new * (1 - mask) + prev_state * mask  # update when meeting 2

            prev_h[0] = h_masked * (1 - state_mask) + prev_h[0] * state_mask  # update when num_seq is even
            prev_h[1] = h_masked * state_mask + prev_h[1] * (1 - state_mask)  # update when num_seq is odd
            return prev_h, num_seq + 1 - mask

    """
    prev_h[0]: word-level last state
    prev_h[1]: hier last state
    prev_h[2]: decoder last state
    prev_h[3]: num_seq
    sphier encoder-decoder model
    """

    def run(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels[0])
        rolled_mask = self.gen_mask(input_labels[1])
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        h_s, num_seq = self.hier_level_rnn(prev_h[1], h, mask, prev_h[3])
        embedding *= mask  # mark first embedding as 0
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(prev_h[2], tf.concat(1, [tf.concat(1, h_s), embedding]), mask)
        return [h, h_s, d, num_seq]

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        num_seq = tf.zeros([self.batch_size, 1])  # id of the current speech, increase when meeting 0 or 2
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_hier = [tf.zeros([self.batch_size, self.c_size]), tf.zeros([self.batch_size, self.c_size])]
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, h_s, h_d, _ = tf.scan(self.run, [self.labels, self.rolled_label],
                               initializer=[init_encode, init_hier, init_decoder, num_seq])
        return [h_s,h_d]

    def decode_bs(self, h_d):
        last_h = h_d[0][-1]
        last_d = h_d[1][-1]
        last_h = tf.concat(1, last_h[:2])
        k = 0
        prev = tf.reshape(last_d, [1, self.h_size])
        prev_h = tf.tile(last_h, [self.beam_size, 1])
        prev_d = tf.tile(last_d, [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev    
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                _, d_new = self.decodernet(tf.concat(1, [prev_h, inp]), prev_d)
                prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        #decoded =  tf.reshape(self.beam_symbols, [self.beam_size, -1])
        return decoded 
 
