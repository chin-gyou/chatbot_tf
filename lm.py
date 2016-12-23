from base import *


class lm(base_enc_dec):
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode, beam_size=5):
        base_enc_dec.__init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode,
                              beam_size)

    """
    prev_h: word-level last state
    language model
    """

    def run(self, prev_h, input_labels):
        embedding = self.embed_labels(input_labels)
        h = self.word_level_rnn(prev_h, embedding)
        return h

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        h_d = tf.scan(self.run, self.labels, initializer=init_encode)
        return [1, h_d]
