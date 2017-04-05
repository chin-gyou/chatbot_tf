from base import *
from initialize import *


class hred(base_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, c_size, vocab_size, embedding, batch_size, learning_rate, mode,beam_size = 5, bi=0):
        self.c_size = c_size
        with tf.variable_scope('hier'):
            self.hiernet = rnn_cell.GRUCell(c_size)
            print(self.context_len)
            self.init_W = tf.get_variable('Init_W',
                                          initializer=tf.random_normal([self.context_len, h_size], stddev=0.01))
            self.init_b = tf.get_variable('Init_b', initializer=tf.zeros([h_size]))
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size,bi)

    @property
    def context_len(self):
        return self.c_size

    """
    word-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, output initializing state
    prev_h: batch_size*h_size
    input_w: batch_size
    """

    def word_level_rnn(self, prev_h, input_w, mask, bi=1):
        embedding = self.embed_labels(input_w)*mask
        prev_h = prev_h * mask  # mask the fist state as zero
        if bi==0:
            with tf.variable_scope('encode', initializer=orthogonal_initializer()):
                _, h_new = self.encodernet(embedding, prev_h)
        else:
            with tf.variable_scope('encode_reverse', initializer=orthogonal_initializer()) as scope:
                try:
                    _, h_new = self.encodernet_r(embedding, prev_h)
                except ValueError:
                    scope.reuse_variables()
                    _, h_new = self.encodernet_r(embedding, prev_h)
        return h_new

    """
    hier-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, update state, else stay unchange
    prev_h: batch_size*c_size
    input: batch_size*h_size
    """

    def hier_level_rnn(self, prev_h, input_vec, mask):
        with tf.variable_scope('hier', initializer=orthogonal_initializer()):
            _, h_new = self.hiernet(input_vec, prev_h)
            h_masked = h_new * (1 - mask) + prev_h * mask  # update when meeting EOU
            return h_masked

    """
    decode-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 2, use initializing state learned from context
    prev_h: batch_size*c_size
    input: batch_size*(c_size+embed_size)
    """
    def decode_level_rnn(self, prev_h, input_h, mask):
        with tf.variable_scope('decode', initializer=orthogonal_initializer()):
            prev_h = prev_h * mask + tf.tanh(tf.matmul(input_h[:, :self.context_len], self.init_W) + self.init_b) * (
            1 - mask)  # learn initial state from context
            _, h_new = self.decodernet(input_h, prev_h)
            return h_new

    """
    if bi==1, input_labels[0] is word index, input_labels[1] is inversed state and input_labels[2] is rolled word index, else only word index and rolled index
    return the encoder hidden state
    """
    def generate_encode(self,prev_h,input_labels):
        if self.bi==0:
            rolled_mask = self.gen_mask(input_labels[1], EOU) 
            return self.word_level_rnn(prev_h, input_labels[0], rolled_mask,0)
        else:
            rolled_mask = self.gen_mask(input_labels[2], EOU)
            h = self.word_level_rnn(prev_h, input_labels[0], rolled_mask,0)
            return tf.concat(1,[h,input_labels[1]])

    """
    prev_h[0]: word-level last state
    prev_h[1]: hier last state
    prev_h[2]: decoder last state
    hier encoder-decoder model
    if bi==1, input_labels[0] is word index, input_labels[1] is inversed state and input_labels[2] is rolled word index, else only word index and rolled index
    """

    def run(self, prev_h, input_labels):
        word = input_labels[0]
        mask = self.gen_mask(word, EOU)
        embedding = self.embed_labels(word)
        h = self.generate_encode(prev_h[0],input_labels)
        h_s = self.hier_level_rnn(prev_h[1], h, mask)
        embedding*=mask#mark first embedding as 0
        # concate embedding and h_s for decoding
        d = self.decode_level_rnn(prev_h[2], tf.concat(1, [h_s, embedding]), mask)
        return [h[:,:self.h_size], h_s, d]

    def run_word(self, prev_h, input_labels, bi=1):
        r_mask = self.gen_mask(input_labels[1], EOU)
        return self.word_level_rnn(prev_h, input_labels[0], r_mask, bi)

    # input_labels[0]: data
    # input_labels[1]: mask
    # mask utterance as the state of the cloest forward one
    def forward_mask(self, prev_h, input_labels):
        data, mask=input_labels
        mask = tf.reshape(mask, [self.batch_size, 1])
        new_h = prev_h * mask + data * (1 - mask)
        return new_h

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encoder = tf.zeros([self.batch_size, self.h_size])
        init_hier = tf.zeros([self.batch_size, self.c_size])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        if self.bi==0:
            _, h_s, h_d = tf.scan(self.run, [self.labels, self.rolled_label],
                            initializer=[init_encoder, init_hier, init_decoder])
        else:
            r_l = tf.reverse(self.labels,dims=[True, False])
            r_l = tf.concat(0,[EOU * tf.ones([1, self.batch_size], dtype=tf.int64), r_l])
            rolled_r_l = tf.concat(0, [EOU * tf.ones([1, self.batch_size], dtype=tf.int64), r_l[:-1]])
            r_h = tf.scan(self.run_word, [r_l, rolled_r_l], initializer=init_encoder)
            r_h = tf.reverse(r_h, dims=[True, False, False])
            mask = tf.cast(tf.not_equal(r_l, EOU), tf.float32)
            mask = tf.reverse(mask,dims=[True, False])
            r_h = tf.scan(self.forward_mask, [r_h, mask], initializer=r_h[0])
            r_h = tf.concat(0, [tf.reshape(r_h[-1],[1,self.batch_size,self.h_size]),r_h[:-2]])
            _, h_s, h_d = tf.scan(self.run, [self.labels, r_h, self.rolled_label],initializer=[init_encoder, init_hier, init_decoder])
        return [h_s,h_d]

    def decode_bs(self, h_d):
        last_h = h_d[0][-1]
        last_d = h_d[1][-1]
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
 
