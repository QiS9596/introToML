import tensorflow as tf


class MultilayerPerceptron:
    def __init__(self, inputsize, output_size, hidden_layer, transfer_function=tf.nn.sigmoid,
                 hidden_layer_activation=None,
                 optimizer=tf.train.AdamOptimizer(0.05), training_scale=0.1):
        """

        :param inputsize: input dimension
        :param output_size: output dimension
        :param hidden_layer: shape of hidden layer, should be n length positive integer array
        :param transfer_function: transfer function
        :param optimizer: optimizer
        :param training_scale: scale of training
        """
        self.sess = tf.Session()
        self.inputsize = inputsize
        self.outputsize = output_size
        self.hidden_layer = hidden_layer
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.training_scale = training_scale
        self.hidden_layer_activation = hidden_layer_activation


        self.layers = []  # layer objects of the MLP
        # add layers
        self.addLayers()

        self.labels = tf.placeholder(tf.float32, shape=[None, self.outputsize])
        self.error = tf.losses.mean_squared_error(self.layers[-1], self.labels)
        self.train_step = self.optimizer.minimize(self.error)
        self.output = self.layers[-1]
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def addLayers(self):
        """
        this function reads the information and build the layer architecture
        interprets the information in inputsize, outputsize and hidden layer array(interger array used to describe the nodes of hidden layers
        the generated layers should be add to self.layers array
        :return:
        """
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

    def predict(self, input):
        self.sess.run(self.output, feed_dict={self.input: input})

    def single_step_train(self, input_data, label):
        self.sess.run(self.train_step, feed_dict={self.input: input_data, self.labels: label})

    def getErrorRate(self, input_data, label):
        return self.sess.run(self.error, feed_dict={self.input: input_data, self.labels: label})

    def epoch_train(self, epoch=20, input_data=None, label=None, fullData=None):
        if (not fullData is None) and input_data is None and label is None:
            input_data = []
            label = []
            for i in fullData:
                input_data.append(i[:-1])
                label.append(i[-1])

        for i in range(0, epoch):
            self.single_step_train(input_data, label)
        return self.getErrorRate(input_data, label)


MLP = MultilayerPerceptron(784, 10, [])
MLP.printLayers()
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

for i in range(0,1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    print(MLP.epoch_train(input_data=batch_xs,label=batch_ys))