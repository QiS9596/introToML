import tensorflow as tf


class MultilayerPerceptron:
    def __init__(self, inputsize, output_size, hidden_layer, transfer_function=tf.nn.sigmoid,
                 hidden_layer_activation=None,
                 optimizer=tf.train.AdamOptimizer(), training_scale=0.1):
        """

        :param inputsize: input dimension
        :param output_size: output dimension
        :param hidden_layer: shape of hidden layer, should be n length positive integer array
        :param transfer_function: transfer function
        :param optimizer: optimizer
        :param training_scale: scale of training
        """
        self.inputsize = inputsize
        self.outputsize = output_size
        self.hidden_layer = hidden_layer
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.transfer_scale = training_scale
        self.hidden_layer_activation = hidden_layer_activation

        self.layers = []  # layer objects of the MLP
        # add layers
        self.addLayers()

    def addLayers(self):
        # adding input layer
        inputlayer_output = self.outputsize
        if len(self.hidden_layer) != 0:
            inputlayer_output = self.hidden_layer[0]
        self.input = tf.placeholder(tf.float32, shape=[None, self.inputsize])
        self.layers.append(self.add_layer(self.inputsize, inputlayer_output, self.input, self.hidden_layer_activation))
        # adding first hidden layer
        if len(self.hidden_layer) != 0:
            first_hiddenlayer_output = self.outputsize
            if len(self.hidden_layer) > 1:
                first_hiddenlayer_output = self.hidden_layer[1]
            self.layers.append(self.add_layer(self.hidden_layer[0], first_hiddenlayer_output, self.layers[0],
                                              self.hidden_layer_activation))

        # adding the rest fo the hidden layer and the output layer
        for i in range(1, len(self.hidden_layer) - 1):
            self.layers.append(self.add_layer(self.hidden_layer[i - 1], self.hidden_layer[i], self.layers[-1],
                                              self.hidden_layer_activation))
        if len(self.hidden_layer) > 1:
            self.layers.append(
                self.add_layer(self.hidden_layer[-1], self.outputsize, self.layers[-1], self.transfer_function))

    def add_layer(self, input_n, output_n, input_tensor, activation_function=None):

        weights = tf.Variable(name='weight', initial_value=tf.random_normal([input_n, output_n]))
        bias = tf.Variable(name='bias', initial_value=tf.random_normal([1, output_n]))
        # print(input_tensor)
        # print(weights)
        layer = tf.add(tf.matmul(input_tensor, weights), bias)
        if activation_function is None:
            output = layer
        else:
            output = activation_function(layer)
        return output

    def printLayers(self):
        for i in self.layers:
            print(i)


MLP = MultilayerPerceptron(2, 1, [3,8])
MLP.printLayers()
