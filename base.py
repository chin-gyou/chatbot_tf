import functools
from tensorflow.python.ops import rnn_cell
import tensorflow as tf


# force every function to execute only once
def exe_once(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# ensure optimise is executed at last so that the graph can be initialised
def init_final(function):
    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        function(self, *args, **kwargs)
        # In test mode, only build graph by prediction, no optimise part
        if self.mode >= 1:
            self.prediction
        elif self.mode ==0: # train mode, optimise
            self.optimise

    return decorator


class base_enc_dec:
    """
    labels: vocab index labels, max_length*batch_size, padding labels are 0s, 2 is eot sign
    length: Number of token of each dialogue batch_size
    embedding: vocab_size*embed_size
    """
    @init_final
    def __init__(self, labels, length, h_size, vocab_size, embedding, batch_size, learning_rate, mode):
        self.__dict__.update(locals())
        self.rolled_label = tf.concat(0, [tf.zeros([1, batch_size], dtype=tf.int32), labels[:-1]])
        with tf.variable_scope('encode'):
            self.encodernet = rnn_cell.GRUCell(h_size)
            # embedding matrix
            self.embedding_W = tf.get_variable('Embedding_W', initializer=embedding)
            self.embedding_b = tf.get_variable('Embedding_b', initializer=tf.zeros([300]))
        with tf.variable_scope('decode'):
            self.decodernet = rnn_cell.GRUCell(h_size)
            self.output_W = tf.get_variable('Output_W', initializer=tf.random_normal([h_size, vocab_size]))
            self.output_b = tf.get_variable('Output_b', initializer=tf.zeros([vocab_size]))

    """
    word-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 0 or 2, output initializing state
    prev_h: batch_size*h_size
    input: batch_size*embed_size
    """

    def word_level_rnn(self, prev_h, input_embedding, mask):
        with tf.variable_scope('encode'):
            prev_h = prev_h * mask  # mask the fist state as zero
            _, h_new = self.encodernet(input_embedding, prev_h)
            return h_new

    """
    decode-level rnn step
    takes the previous state and new input, output the new hidden state
    If meeting 0 or 2, output initializing state
    prev_h: batch_size*h_size
    input: batch_size*h_size
    """

    def decode_level_rnn(self, prev_h, input_h, mask):
        with tf.variable_scope('decode'):
            prev_h = prev_h * mask  # mask the fist state as zero
            _, h_new = self.decodernet(input_h, prev_h)
            return h_new

    """
    prev_h[0]: word-level last state
    prev_h[1]: decoder last state
    basic encoder-decoder model
    """

    def run(self, prev_h, input_labels):
        rolled_mask = self.gen_mask(input_labels[1])
        embedding = self.embed_labels(input_labels[0])
        h = self.word_level_rnn(prev_h[0], embedding, rolled_mask)
        d = self.decode_level_rnn(prev_h[1], h, rolled_mask)
        return [h, d]

    # turn labels into corresponding embeddings
    def embed_labels(self, input_labels):
        return tf.gather(self.embedding_W, input_labels) + self.embedding_b  #embedded inputs, batch_size*embed_size

    # generate mask for label, batch_size*1
    def gen_mask(self, input_labels):
        # mask all 0 and 2 as 0
        mask = tf.cast(tf.logical_and(input_labels > 0, tf.not_equal(input_labels, 2)), tf.float32)
        return tf.reshape(mask, [self.batch_size, 1])

    # scan step, return output hidden state of the output layer
    # h_d states after running, max_len*batch_size*h_size
    def scan_step(self):
        init_encode = tf.zeros([self.batch_size, self.h_size])
        init_decoder = tf.zeros([self.batch_size, self.h_size])
        _, h_d = tf.scan(self.run, [self.labels, self.rolled_label], initializer=[init_encode, init_decoder])
        return h_d

    # return output layer
    @exe_once
    def prediction(self):
        h_d = self.scan_step()
        predicted = tf.reshape(h_d[:-1], [-1, self.h_size])  # exclude the last prediction
        output = tf.matmul(predicted, self.output_W) + self.output_b  #((max_len-1)*batch_size)*vocab_size
        return output

    @exe_once
    def cost(self):
        y_flat = tf.reshape(self.labels[1:], [-1])  # exclude the first label
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction,y_flat)
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = tf.reshape(mask * loss, tf.shape(self.labels[1:]))
        # normalized loss per example
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=0) / tf.to_float(self.length)
        return tf.reduce_mean(mean_loss_by_example)  # average loss of the batch

    @exe_once
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost, global_step=global_step)
        return global_step, train_op