from hred_enc_dec import *
from dataproducer import *

s1 = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
inputs = np.array([s1] * 5, dtype=np.float32)
s_labels = np.array([[1, 2, 3], [1, 2, 3]])
labels = np.array([s_labels] * 5)
length = np.array([[3] * 2] * 5)


# hierarchical seq2seq model with separated lines
class sphred_enc_dec(hred_enc_dec):
    # c_size: hidden layer size of sentence-level RNN
    @init_final
    def __init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size, embedding,learning_rate,
                 decoded=1,mode=0):
        hred_enc_dec.__init__(self, data, labels, length, h_size, e_size, c_size, batch_size, num_seq, vocab_size,
                              embedding, learning_rate,decoded,mode)

    # input of context for decoding sequences
    def _context_input(self, h, i, max_len):
        context = tf.concat(concat_dim=1, values=[h[i], h[i + 1]])
        context_input = tf.tile(context, [max_len, 1])
        return tf.reshape(context_input, [tf.shape(context)[0], max_len, 2 * self.c_size])

    # sequence-level encoding, only decode the last sequence, deprecated
    def __encode_sequence_last(self, h):
        # encode in sentence-level
        with tf.variable_scope('hierarchical') as hier:
            states_A = tf.transpose(tf.pack(h[::2]), perm=[1, 0, 2])  # batch_size*num*encoder_state
            states_B = tf.transpose(tf.pack(h[1::2]), perm=[1, 0, 2])
            sentence = rnn_cell.GRUCell(self.c_size)
            _, h_stateA = rnn.dynamic_rnn(sentence, states_A, dtype=tf.float32)
            hier.reuse_variables()
            _, h_stateB = rnn.dynamic_rnn(sentence, states_B, dtype=tf.float32)
            return h_stateA, h_stateB

    # sequence-level encoding, keep needed states for {decoded} last sequences
    # e is word-level hidden states
    def encode_sequence(self, e):
        # h_state[i] keeps context state for the (i+1)th sequence, with an additional initializing context
        # when no need for decoding, also assigned as zeros
        h_state = [tf.zeros([self.batch_size, self.c_size])] * (self.num_seq - self.decoded - 1)
        # encode in sentence-level
        with tf.variable_scope('hierarchical') as hier:
            initial_stateA, initial_stateB = tf.zeros([self.batch_size, self.c_size]), tf.zeros(
                [self.batch_size, self.c_size])
            # run sequence-level rnn until decoding is needed, these states don't need to be stored
            if self.num_seq - self.decoded > 2:
                _, initial_stateA = rnn.dynamic_rnn(self.hred, tf.transpose(e[1:self.num_seq - self.decoded - 1:2],
                                                                            perm=[1, 0, 2]),
                                                    dtype=tf.float32, initial_state=initial_stateA)
                hier.reuse_variables()
                if self.num_seq - self.decoded > 3:
                    _, initial_stateB = rnn.dynamic_rnn(self.hred, tf.transpose(e[2:self.num_seq - self.decoded - 1:2],
                                                                                perm=[1, 0, 2]),
                                                        dtype=tf.float32, initial_state=initial_stateB)
            # run senquence-level rnn while keeping all the hidden states
            for i in range(max(self.num_seq - self.decoded - 2, 0), self.num_seq):
                if i > max(self.num_seq - self.decoded - 2, 0):
                    hier.reuse_variables()
                if i % 2 == 0:
                    _, initial_stateA = rnn.dynamic_rnn(self.hred,
                                                        tf.reshape(e[i + 1], [self.batch_size, 1, self.h_size]),
                                                        dtype=tf.float32,
                                                        initial_state=initial_stateA)
                    h_state.append(initial_stateA)
                else:
                    _, initial_stateB = rnn.dynamic_rnn(self.hred,
                                                        tf.reshape(e[i + 1], [self.batch_size, 1, self.h_size]),
                                                        dtype=tf.float32, initial_state=initial_stateB)
                    h_state.append(initial_stateB)
            return h_state

    # only decode last sequence, deprecated
    def __decode_last(self, h_a, h_b, e):
        # decode
        with tf.variable_scope('decode') as dec:
            context = tf.concat(1, [h_a, h_b])
            max_len = tf.shape(self.data[-1])[2]
            context_input = tf.tile(context, [max_len, 1])
            context_input = tf.reshape(context_input, [tf.shape(context)[0], max_len, 2 * self.h_size])
            response = rnn_cell.GRUCell(self.h_size)
            output, _ = rnn.dynamic_rnn(response, tf.concat(2, [self.data[-1], context_input]),
                                        sequence_length=self.length[-1], dtype=tf.float32,
                                        initial_state=e[-2])
            W = tf.Variable(tf.zeros([response.output_size, self.vocab_size]), dtype=tf.float32, name='Output_W')
            b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='Output_b')

            logits_flat = tf.matmul(tf.reshape(output, [-1, response.output_size]), W) + b
            return logits_flat

    def decode_forward(self,h, sentence_input, initial):
        i = self.num_seq - 2

        with tf.variable_scope('decode') as dec:
            context = tf.concat(concat_dim=1, values=[h[i], h[i + 1]]) # h_size*2*c_size
            context_input = tf.reshape(context, [tf.shape(context)[0], 1, 2 * self.c_size])
            sentence_input = tf.reshape(sentence_input, [-1,self.vocab_size])  # data[i] batch_size*max_length*feature_size
            embedded = tf.matmul(sentence_input,self.W) + self.b
            output, dec_state = rnn.dynamic_rnn(self.decodernet, tf.concat(2, [tf.reshape(embedded,[1,1,300]), context_input]),
                                       sequence_length=[1], dtype=tf.float32, initial_state=initial)
            W = tf.Variable(tf.zeros([self.decodernet.output_size, self.vocab_size]), dtype=tf.float32,
                             name='Output_W')
            b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='Output_b')
            word_ind = tf.argmax(tf.nn.softmax(tf.matmul(tf.reshape(output,[-1,self.decodernet.output_size]),W) + b),1)
            return word_ind, dec_state


if __name__ == '__main__':
    # batch_size=20
    # num_sequence=3
    # data_producer=dataproducer(1,2,3,num_sequence=3)
    # labels=tf.placeholder(tf.float32, [num_sequence,batch_size, None])
    t = sphred_enc_dec(data=inputs, labels=labels, length=length, h_size=2, e_size=1, c_size=2, batch_size=2, num_seq=5,
                       vocab_size=8, embedding=2)
    # sess = tf.Session()
    # sess.run(tf.initialize_all_variables())
    # w = sess.run(t.cost)
    # print(w)
    # print(np.shape(inputs[0]))
