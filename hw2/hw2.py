import tensorflow as tf


class MultilayerPerceptron:
    def __init__(self, inputsize, output_size, hidden_layer, transfer_function=tf.nn.sigmoid,
                 optimizer=tf.train.AdamOptimizer(), training_scale=0.1):
        self.inputsize = inputsize
        self.outputsize = output_size
        self.hidden_layer = hidden_layer
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.transfer_scale = training_scale

    def add_layer(self, input_n, output_n, activation_function= None):
        # weights = tf.Variable(name = 'weight',initial_value=)
        pass