import functools
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn


# force every function to execute only once
def define_scope(function):
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
    vocab_size should contain an additional 0 class for padding
    data: sequence data, num_sentences*batch_size*max_length*feature_size
    labels: vocab index labels, num_sentences*batch_size*max_length, padding labels are 0s
    length: length of every sequence, num_sequence*batch_size
    h_size: hidden layer size of word-level RNN
    e_size: embedding vector size for one word
    embedding: initialising embedding matrix, vocab_size*e_size
    decoded: number of decoded senquences, by default only decode the last sequence
    bn: whether batch normalisation is used for context concatenation
    base_rnn: whether put the ending state of a sequence as the starting state of the next one, defautl not
    """

    @init_final
    def __init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding, learning_rate,
                 decoded=1, mode=0, bn=0, base_rnn=0):
        self.__dict__.update(locals())
        self.W = tf.Variable(self.embedding, name='Embedding_W')
        self.b = tf.Variable(tf.zeros([self.e_size]), dtype=tf.float32, name='Embedding_b')

        with tf.variable_scope('encode'):
            self.encodernet = rnn_cell.GRUCell(self.h_size)
        with tf.variable_scope('decode'):
            self.decodernet = rnn_cell.GRUCell(self.h_size)
            # mapping to vocab probability
            self.W2 = tf.Variable(tf.zeros([self.decodernet.output_size, self.vocab_size]), dtype=tf.float32,
                                  name='Output_W')
            self.b2 = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='Output_b')

    # start word of decoded sequence
    def start_word(self):
        return tf.zeros([self.batch_size, 1, self.vocab_size])

    def embedded_word(self, v, shape):
        reshaped = tf.reshape(v, [-1, self.vocab_size])
        embedded = tf.matmul(reshaped, self.W) + self.b
        return tf.reshape(embedded, shape)

    # input of last word for decoding sequences
    def _sentence_input(self, i, max_len):
        sentence_input = tf.slice(self.data[i + 1], [0, 0, 0], [self.batch_size, max_len - 1, self.vocab_size])
        sentence_input = tf.concat(1, [self.start_word(), sentence_input])
        shape = tf.shape(self.data[i + 1])
        return self.embedded_word(sentence_input, [shape[0], shape[1], 300])

    # encode in word-level, return a list of encode_states for the first {num_seq-1} sequences
    # encoder_state[i]: the encoder_state of the (i-1)-th sentence, shape=batch_size*h_state, the first one is zero initialisation
    def encode_word(self):
        encoder_states = [tf.zeros([self.batch_size, self.h_size])]  # zero-initialisation for the first state

        # encode in word-level
        with tf.variable_scope('encode') as enc:
            initial_state = tf.zeros([self.batch_size, self.h_size])
            for i in range(self.num_seq):
                concatenated = tf.reshape(self.data[i], [-1, self.vocab_size])
                embedded = tf.matmul(concatenated, self.W) + self.b
                shape = tf.shape(self.data[i])
                _, encoder_state = rnn.dynamic_rnn(self.encodernet, tf.reshape(embedded, [shape[0], shape[1], 300]),
                                                   sequence_length=self.length[i], dtype=tf.float32,
                                                   initial_state=initial_state)
                enc.reuse_variables()
                if self.base_rnn:
                    initial_state = encoder_state
                encoder_states.append(encoder_state)
        return encoder_states

    def decode(self, e):
        decoded = []
        with tf.variable_scope('decode') as dec:
            # decode, starts from the context state before the first decoded sequence
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                if self.mode < 3:
                    max_len = tf.shape(self.data[i + 1])[1]
                    output, _ = rnn.dynamic_rnn(self.decodernet, self._sentence_input(i, max_len),
                                                sequence_length=self.length[i + 1], dtype=tf.float32,
                                                initial_state=e[i + 1])
                    dec.reuse_variables()
                    # output: batch_size*max_length*h_size
                    decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
        return decoded

    # prediction runs the forward computation
    # the output should be of size {decoded*batch_size*max_length}*vocab_size
    @define_scope
    def prediction(self):
        encoder_states = base_enc_dec.encode_word(self)
        output = self.decode(encoder_states)
        return output

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

    # compute the mean cross entropy for the last but i sequence
    def mean_cross_entropy(self, y_flat, losses, i):
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses
        # Bring back to [batch, max_length] shape
        masked_losses = tf.reshape(masked_losses, tf.shape(self.labels[-i]))

        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(
            self.length[-i])
        mean_loss = tf.reduce_mean(mean_loss_by_example)
        return mean_loss

    # cost for training
    # cost computes the loss when decoding the specified {decoded} sequences
    # returned loss is the mean loss for every decoded sequence
    @define_scope
    def cost(self):
        total_loss = 0
        for i in range(1, self.decoded + 1):
            y_flat = tf.reshape(self.labels[-i], [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.prediction[-i], y_flat)
            total_loss += self.mean_cross_entropy(y_flat, losses, i)
        return total_loss / self.decoded

    # percentage of predicted word in the top-k list
    # averaged for {decoded} sequences
    def top_k_per(self, k):
        total_loss = 0
        for i in range(1, self.decoded + 1):
            y_flat = tf.reshape(self.labels[-i], [-1])
            losses = tf.cast(tf.nn.in_top_k(self.prediction[-i], y_flat, k), tf.float32)
            total_loss += self.mean_cross_entropy(y_flat, losses, i)
        return total_loss / self.decoded

    @define_scope
    def optimise(self):
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost, global_step=global_step)
        return global_step, train_op
