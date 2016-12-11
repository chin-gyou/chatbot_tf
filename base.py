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
        elif self.mode == 0:  # train mode, optimise
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
    """

    @init_final
    def __init__(self, data, labels, length, h_size, e_size, batch_size, num_seq, vocab_size, embedding, learning_rate,
                 decoded=1, mode=0, bn=0, beam_size=5):
        self.__dict__.update(locals())
        self.W = tf.Variable(self.embedding, name='Embedding_W')
        self.b = tf.Variable(tf.zeros([self.e_size]), dtype=tf.float32, name='Embedding_b')
        self.log_beam_probs, self.beam_path, self.output_beam_symbols, self.beam_symbols = [], [], [], []
        # with tf.variable_scope('encode'):
        #    self.encodernet = rnn_cell.GRUCell(self.c_size)
        #    self.encodebw = rnn_cell.GRUCell(self.h_size)
        with tf.variable_scope('decode'):
            self.decodernet = rnn_cell.GRUCell(self.h_size)
            # self.decW = tf.Variable(tf.zeros([self., self.vocab_size]), dtype=tf.float32,
            #                      name='decode_init_W')
            # self.decb = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32, name='decode_init_b')
            # mapping to vocab probability
            self.W2 = tf.Variable(tf.random_normal([self.decodernet.output_size, self.vocab_size]),
                                  name="Output_W")
            # self.W2 = tf.Variable(tf.zeros([self.decodernet.output_size, self.vocab_size]), dtype=tf.float32,
            #                      name='Output_W')
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
        outputs = []
        # encode in word-level
        with tf.variable_scope('encode') as enc:
            initial_state = tf.zeros([self.batch_size, self.h_size])
            for i in range(self.num_seq):
                concatenated = tf.reshape(self.data[i], [-1, self.vocab_size])
                embedded = tf.matmul(concatenated, self.W) + self.b
                shape = tf.shape(self.data[i])
                _, encoder_state = tf.nn.dynamic_rnn(self.encodernet, tf.reshape(embedded, [shape[0], shape[1], 300]),
                                                     sequence_length=self.length[i], dtype=tf.float32,
                                                     initial_state=initial_state)
                enc.reuse_variables()
                initial_state = encoder_state
                # concatenated_state=tf.concat(1,encoder_state)
                # print(output)
                # max_len = tf.shape(self.data[i])[1]
                # outputs.append(tf.matmul(tf.reshape(tf.slice(output,[0,0,0],[self.batch_size, max_len, self.encodernet.output_size]), [-1, self.encodernet.output_size]), self.W2) + self.b2)
                encoder_states.append(encoder_state)
        return encoder_states

    def decode(self, e):
        decoded = []
        with tf.variable_scope('decode') as dec:
            # decode, starts from the context state before the first decoded sequence
            for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
                if self.mode < 2:
                    max_len = tf.shape(self.data[i + 1])[1]
                    output, _ = rnn.dynamic_rnn(self.decodernet, self._sentence_input(i, max_len),
                                                sequence_length=self.length[i + 1], dtype=tf.float32,
                                                initial_state=e[i + 1])
                    dec.reuse_variables()
                    # output: batch_size*max_length*h_size
                    decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
                if self.mode == 1:
                    k = 0
                    inp = self.embedded_word(self.start_word(), [1, 1, 300])
                    initial_state = e[i + 1]
                    state = initial_state
                    outputs = []
                    prev = None
                    state_size = int(initial_state.get_shape().with_rank(2)[1])
                    while k < 12:
                        if prev is not None:
                            inp = self.beam_search(prev, k)
                            shape = inp.get_shape()
                            inp = tf.reshape(inp, [int(shape[0]), 1, int(shape[1])])
                        if k > 0:
                            dec.reuse_variables()
                        length = 1 if k == 0 else self.beam_size
                        output, state = rnn.dynamic_rnn(self.decodernet,
                                                        inp,
                                                        dtype=tf.float32, initial_state=state)

                        prev = output
                        if k == 0:
                            state = tf.tile(state, [self.beam_size, 1])
                        k += 1
                    decoded_sequence = tf.reshape(self.output_beam_symbols[-1], [self.beam_size, -1])
        return decoded if self.mode < 1 else decoded_sequence

    def beam_search(self, prev, k):
        probs = tf.log(tf.nn.softmax(
            tf.matmul(tf.reshape(prev, [-1, self.decodernet.output_size]), self.W2) + self.b2))
        minus_probs = [0 for i in range(self.vocab_size)]
        minus_probs[1] = -1e20
        minus_probs[2] = -1e20
        # minus_probs[10963] = -1e20
        # minus_probs[428] = -1e20
        probs = probs + minus_probs
        if k > 1:
            probs = tf.reshape(probs + self.log_beam_probs[-1],
                               [-1, self.beam_size * self.vocab_size])

        best_probs, indices = tf.nn.top_k(probs, self.beam_size)
        indices = tf.reshape(indices, [-1, 1])
        best_probs = tf.reshape(best_probs, [-1, 1])

        symbols = indices % self.vocab_size  # Which word in vocabulary.
        beam_parent = indices // self.vocab_size  # Which hypothesis it came from.

        self.beam_path.append(beam_parent)
        symbols_live = symbols
        if k > 1:
            symbols_history = tf.gather(self.output_beam_symbols[-1], beam_parent)
            symbols_live = tf.concat(1, [tf.reshape(symbols_history, [-1, k - 1]), tf.reshape(symbols, [-1, 1])])
        self.output_beam_symbols.append(symbols_live)
        self.beam_symbols.append(symbols)
        self.log_beam_probs.append(best_probs)

        concatenated = tf.reshape(
            tf.one_hot(symbols, depth=self.vocab_size, dtype=tf.float32), [-1, self.vocab_size])
        embedded = tf.matmul(concatenated, self.W) + self.b
        emb_prev = tf.reshape(embedded, [self.beam_size, 300])
        return emb_prev

    # def decode(self,e):
    #    decoded=[]
    #    with tf.variable_scope('decode') as dec:
    #        # decode, starts from the context state before the first decoded sequence
    #        for i in range(self.num_seq - self.decoded - 1, self.num_seq - 1):
    #            if self.mode < 3:
    #                max_len = tf.shape(self.data[i + 1])[1]
    #                output, _ = rnn.dynamic_rnn(self.decodernet, self._sentence_input(i,max_len),
    #                                            sequence_length=self.length[i + 1], dtype=tf.float32, initial_state=e[i+1])
    #                dec.reuse_variables()
    #                # output: batch_size*max_length*h_size
    #                decoded.append(tf.matmul(tf.reshape(output, [-1, self.decodernet.output_size]), self.W2) + self.b2)
    #    return decoded

    # prediction runs the forward computation
    # the output should be of size {decoded*batch_size*max_length}*vocab_size
    @define_scope
    def prediction(self):
        encoder_states = base_enc_dec.encode_word(self)
        dec_output = self.decode(encoder_states)
        return dec_output

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
        # weight=tf.linspace(2.0, 0.8, tf.shape(self.labels[-i])[1])
        masked_losses = tf.reshape(masked_losses, tf.shape(self.labels[-i]))
        # weighted_losses = tf.mul(masked_losses,weight)
        # return masked_losses
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
        dec = self.prediction
        total_loss = 0
        for i in range(1, self.decoded + 1):
            y_flat = tf.reshape(self.labels[-i], [-1])
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(dec[-i], y_flat)
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
        # return self.cost
        optim = tf.train.AdamOptimizer(self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optim.minimize(self.cost, global_step=global_step)
        return global_step, train_op
