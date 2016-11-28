import tensorflow as tf

# Fully-connected layer
class Dense():

    def __init__(self, scope="dense_layer", size=None, dropout=1.,
                 nonlinearity=tf.identity):
        assert size, "Must specify layer size (num nodes)"
        self.__dict__.update(locals())

    # x: batch_size*x_size
    # return: batch_size*self.size
    def __call__(self, x, x_size):
        with tf.name_scope(self.scope):
            while True:
                try:  # reuse weights if already initialized
                    reshaped=tf.reshape(x,[-1,x_size])
                    result=self.nonlinearity(tf.matmul(reshaped, self.w) + self.b)
                    return tf.reshape(result,[tf.shape(x)[0],self.size])
                except(AttributeError):
                    self.w, self.b = self.wbVars(x.get_shape()[1].value, self.size)
                    self.w = tf.nn.dropout(self.w, self.dropout)

    # Helper to initialize weights and biases, via He's adaptation
    # of Xavier init for ReLUs: https://arxiv.org/abs/1502.01852
    @staticmethod
    def wbVars(fan_in, fan_out):

        stddev = tf.cast((2 / fan_in) ** 0.5, tf.float32)

        initial_w = tf.random_normal([fan_in, fan_out], stddev=stddev)
        initial_b = tf.zeros([fan_out])

        return (tf.Variable(initial_w, trainable=True, name="weights"),
                tf.Variable(initial_b, trainable=True, name="biases"))
