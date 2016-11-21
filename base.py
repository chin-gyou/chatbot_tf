import functools
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn


# force every function to execute only once
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# ensure optimise is executed at last so that the graph can be initialised
def init_final(function):
    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        function(self, *args, **kwargs)
        self.optimise

    return decorator


class base_enc_dec:
    """
    vocab_size should contain an additional 0 class for padding
    data: sequence data, num_sentences*batch_size*max_length*feature_size
    labels: vocab index labels, num_sentences*batch_size*max_length, padding labels are 0s
    length: length of every sequence, num_sequence*batch_size
    h_size: hidden layer size of word-level RNN
    e_size: embedding vector size for one word
    embedding: initialising embedding matrix, vocab_size*e_size
    decoded: number of decoded senquences, by default only decode the last sequence
    """

    def __init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding, learning_rate, decoded=1, mode=0):
        self.__dict__.update(locals())
        self.W = tf.Variable(self.embedding, name='Embedding_W')
        self.b = tf.Variable(tf.zeros([self.e_size]), dtype=tf.float32, name='Embedding_b')
        with tf.variable_scope('encode'):
            self.encodernet = rnn_cell.GRUCell(self.h_size)
        with tf.variable_scope('decode'):
            self.decodernet = rnn_cell.GRUCell(self.h_size)

    # input of last word for decoding sequences
    def _sentence_input(self, i, max_len):
        sentence_input = tf.slice(self.data[i + 1], [0, 1, 0], [self.batch_size, max_len - 1, self.vocab_size])
        sentence_input = tf.concat(1, [tf.zeros([self.batch_size, 1, self.vocab_size]), sentence_input])
        concatenated = tf.reshape(sentence_input, [-1, self.vocab_size])
        embedded = tf.matmul(concatenated, self.W) + self.b
        shape = tf.shape(self.data[i + 1])
        return tf.reshape(embedded, [shape[0], shape[1], 300])

    # encode in word-level, return a list of encode_states for the first {num_seq-1} sequences
    # encoder_state[i]: the encoder_state of the (i+1)-th sentence, shape=batch_size*h_state, the first one is zero initialisation
    def encode_word(self):
        encoder_states = [tf.zeros([self.batch_size, self.h_size])]  # zero-initialisation for the first state

        # encode in word-level
        with tf.variable_scope('encode') as enc:
            for i in range(self.num_seq):
                concatenated = tf.reshape(self.data[i], [-1, self.vocab_size])
                embedded = tf.matmul(concatenated, self.W) + self.b
                shape = tf.shape(self.data[i])
                _, encoder_state = rnn.dynamic_rnn(self.encodernet, tf.reshape(embedded, [shape[0], shape[1], 300]),
                                                   sequence_length=self.length[i], dtype=tf.float32)
                enc.reuse_variables()
                encoder_states.append(encoder_state)
        return encoder_states

    # prediction runs the forward computation
    # the output should be of size {decoded*batch_size*max_length}*vocab_size
    @define_scope
    def prediction(self):
        pass

    @define_scope
    # loss when decoding only the last sequence, already deprecated
    def __cost_last(self, logits_flat):
        y_flat = tf.reshape(self.labels[-1], [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_flat, y_flat)
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses
        # Bring back to [B, T] shape
        masked_losses = tf.reshape(masked_losses, tf.shape(self.labels[-1]))

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / self.length[-1]
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        return mean_loss

    def mean_cross_entropy(self, y_flat, losses):
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses
        # Bring back to [decoded, batch, max_length] shape
        masked_losses = tf.reshape(masked_losses, tf.shape(self.labels[-self.decoded:]))

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=2) / tf.to_float(
            self.length[-self.decoded:])
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        return mean_loss

    # cost computes the loss when decoding the specified {decoded} sequences
    # returned loss is the mean loss for every decoded sequence
    @define_scope
    def cost(self):
        y_flat = tf.reshape(self.labels[-self.decoded:], [-1])
        if self.mode:
            self.prediction = tf.stop_gradient(self.prediction, 'stop_gradients')
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction, y_flat)
        return self.mean_cross_entropy(y_flat, losses)

    @define_scope
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost, global_step=global_step)
        return global_step, train_op
