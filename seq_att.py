from base import *
from dense import *
num = 3
# baseline seq2seq model
class seq_attn(base_enc_dec):
    @init_final
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode, beam_size=5):
        with tf.variable_scope('decode'):
            self.init_W = tf.get_variable('Init_W', initializer=tf.random_normal([h_size, h_size]))
            self.init_b = tf.get_variable('Init_b', initializer=tf.zeros([h_size]))
            self.attn_W1 = tf.get_variable('Attn_W1', initializer=tf.random_normal([1, 1, h_size, h_size]))
            self.attn_V = tf.get_variable('Attn_V', initializer=tf.random_normal([h_size]))
            self.attn_W2 = Dense("Attenion", h_size, h_size, name='Attn_W2')
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    """
    prev_h[0]: word-level last state
    prev_h[1]: decoder last state
    baseline seq2seq model
    """

    def run_encode(self, prev_h, input_labels):
        mask = self.gen_mask(input_labels, EOU)
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h[0], embedding)
  #      prev_h[1] = prev_h[1] * mask + tf.tanh(tf.matmul(h, self.init_W) + self.init_b) * (
  #      1 - mask)  # learn initial state from context
  #      d = self.decode_level_rnn(prev_h[1], embedding)
        return [h, prev_h[1] + 1 - mask]

    def run_decode(self, prev_h, input_labels):
        h, labels, num_seq = input_labels
        mask = self.gen_mask(labels, EOU)
        embedding = self.embed_labels(labels)
        prev_h = prev_h * mask + tf.tanh(tf.matmul(h, self.init_W) + self.init_b) * (
        1 - mask)
        # get needed h
        attn = self.attention(self.h, prev_h, num_seq)
        return self.decode_level_rnn(prev_h, tf.concat(1,[embedding,h,attn]))

        
    # attention_states: attn_length*batch_size*hidden_size
    # d: hidden state of last step for decoding
    def attention(self, attention_states, d, num_seq):
        attn_length = tf.shape(attention_states)[0]
        hidden = tf.reshape(attention_states,[attn_length, self.batch_size, 1, self.h_size])
        # v^T * tanh(w1*h + w2*d)
        hidden_features = tf.nn.conv2d(hidden, self.attn_W1, [1, 1, 1, 1], "SAME") #attn_length*batch_size*1*hidden_size
        y = tf.reshape(self.attn_W2(d), [1, -1, 1, self.h_size])
        s = tf.reduce_sum(self.attn_V * tf.tanh(hidden_features + y), [2, 3])# attn_length*batch_size
        mask = tf.reshape(tf.cast(tf.less(self.num_seq, num_seq), tf.float32), [attn_length, -1])
        print "mask", mask
        a = tf.exp(s)*mask
        a = tf.div(a, tf.reshape(tf.reduce_sum(a,[0]),[1,self.batch_size]))
        # d = sigma(ai*hi)
        new_d = tf.reduce_sum(tf.reshape(a, [attn_length, self.batch_size, 1, 1]) * hidden, [0, 2])
        new_d = tf.reshape(d, [-1, self.h_size])
        return new_d # batch_size*h_size

    def scan_step(self):
        num_seq = tf.zeros([self.batch_size, 1])
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_decode = tf.zeros([self.batch_size, self.h_size])
        self.h, self.num_seq = tf.scan(self.run_encode, self.labels, initializer=[init_encode, num_seq])
        h_d = tf.scan(self.run_decode, [self.h, self.labels, self.num_seq], initializer=init_decode) 
        return [self.h, h_d]


    def decode_bs(self, h_d):
        last_d = h_d[1][-1]
        k = 0
        prev = tf.reshape(last_d, [1, self.h_size])
        prev_d = tf.tile(prev, [self.beam_size, 1])
        while k < 15:
            if k == 0:
                prev_d = prev    
            inp = self.beam_search(prev_d, k)
            prev_d = tf.reshape(tf.gather(prev_d, self.beam_path[-1]), [self.beam_size, self.h_size])
            k += 1
            with tf.variable_scope('decode') as dec:
                dec.reuse_variables()
                _, d_new = self.decodernet(inp, prev_d)
                prev_d = d_new
        decoded =  tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        #decoded =  tf.reshape(self.beam_symbols, [self.beam_size, -1])
        return decoded 
 
    
