import random

import tensorflow as tf


class MultilayerPerceptron:
    def __init__(self, inputsize, output_size, hidden_layer, transfer_function=tf.nn.sigmoid,
                 hidden_layer_activation=None,
                 optimizer=tf.train.GradientDescentOptimizer(2)):
        """

        :param inputsize: input dimension
        :param output_size: output dimension
        :param hidden_layer: shape of hidden layer, should be n length positive integer array
        :param transfer_function: transfer function
        :param optimizer: optimizer
        """
        self.sess = tf.Session()
        self.inputsize = inputsize
        self.outputsize = output_size
        self.hidden_layer = hidden_layer
        self.transfer_function = transfer_function
        self.optimizer = optimizer
        self.hidden_layer_activation = hidden_layer_activation

        self.layers = []  # layer objects of the MLP

        self.labels = tf.placeholder(tf.float32, shape=[None, self.outputsize])
        self.input = tf.placeholder(tf.float32, shape=[None, self.inputsize])  # TODO

        # add layers
        self.addLayers()
        self.printLayers()

        self.error = tf.losses.mean_squared_error( self.labels,self.layers[-1])
        #self.error = tf.keras.backend.binary_crossentropy(target = self.labels, output = self.layers[-1])
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

        self.layers.append(self.add_layer(self.inputsize, inputlayer_output, self.input, self.hidden_layer_activation))
        # adding first hidden layer
        if len(self.hidden_layer) != 0:
            first_hiddenlayer_output = self.outputsize
            if len(self.hidden_layer) > 1:
                first_hiddenlayer_output = self.hidden_layer[1]
            self.layers.append(self.add_layer(self.hidden_layer[0], first_hiddenlayer_output, self.layers[-1],
                                              self.hidden_layer_activation))

        # adding the rest fo the hidden layer and the output layer
        for i in range(1, len(self.hidden_layer) - 1):
            self.layers.append(self.add_layer(self.hidden_layer[i], self.hidden_layer[i + 1], self.layers[-1],
                                              self.hidden_layer_activation))
        if len(self.hidden_layer) > 1:
            self.layers.append(
                self.add_layer(self.hidden_layer[-1], self.outputsize, self.layers[-1], self.transfer_function))

    def add_layer(self, input_n, output_n, input_tensor, activation_function=None):
        with tf.name_scope('layer'):
            with tf.name_scope('wights'):
                weights = tf.Variable(name='weight', initial_value=tf.truncated_normal(shape=[input_n, output_n],mean=0.0,stddev=0.0001),dtype=tf.float32)

            with tf.name_scope('biases'):
                bias = tf.Variable(name='bias', initial_value=tf.zeros([1, output_n]),dtype=tf.float32)

            with tf.name_scope('neuron'):
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
        return self.sess.run(self.output, feed_dict={self.input: input})

    def hitrate(self, input, label):
        hit = 0.
        for i in range(0, len(input)):
            a = self.predict([input[i]])[0]

            # print(input[i])
            # print(a,end=' ')
            # print(str(label[i][0]))
            # print('\n--------\n')

            if a < 0.5:
                a = 0
            else: a = 1

            if a == int(label[i][0]):
                hit += 1
        return hit / float(len(input))

    def single_step_train(self, input_data, label):
        self.sess.run(self.train_step, feed_dict={self.input: input_data, self.labels: label})

    def getErrorRate(self, input_data, label):
        return self.sess.run(self.error, feed_dict={self.input: input_data, self.labels: label})

    def printWeights(self):
        print("printWeight function called")
        for v in tf.trainable_variables():
            print(v.eval(session=self.sess))
        print("-------OuO-------")

    def epoch_train(self, epoch=20, input_data=None, label=None, fullData=None):
        if (not fullData is None) and input_data is None and label is None:
            input_data = []
            label = []
            for i in fullData:
                input_data.append(i[:-1])
                label.append(i[-1])

        for i in range(0, epoch):
            #self.printWeights()
            self.single_step_train(input_data, label)

            #print('loss',self.getErrorRate(input_data, label))
        return self.getErrorRate(input_data, label)


# MLP = MultilayerPerceptron(784, 10, [])
# MLP.printLayers()
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# for i in range(0,1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     print(MLP.epoch_train(input_data=batch_xs,label=batch_ys))


def getHiddenarchitecture(min_layer_number=5, max_layer_number=10, min_node_num=30, max_node_num=300):
    hidden = []
    for i in range(min_layer_number, random.randint(min_layer_number, max_layer_number)):
        hidden.append(random.randint(min_node_num, max_node_num))
    return hidden


def tryOneArchitecture(input_dimension, output_dimension, input_data, labels, hidden):
    MLP = MultilayerPerceptron(input_dimension, output_dimension, hidden, hidden_layer_activation=tf.nn.relu)
    for i in range(10):
        print(MLP.epoch_train(input_data=input_data, label=labels))

    print('test')

    #print(MLP.hitrate(input_data[500:1000], labels[500:1000]))
    return MLP


input_dimension = 6
output_dimension = 1
from xlsReader import excelReader

reader = excelReader('./training data(1000).xlsx')
data, labels = reader.processData()
#print(data.dtype)
#print(labels.dtype)
'''
input_dimension=2
import numpy as np

a=np.array([1,1],dtype=np.float32).reshape(-1,2)
b=np.array([0.5],dtype=np.float32).reshape(-1,1)
MLP = tryOneArchitecture(input_dimension, output_dimension, a, b, [2,2])
'''
MLP = tryOneArchitecture(input_dimension, output_dimension, data, labels, [30,30,30,30])
reader = excelReader('./testing data.xlsx')
data, labels = reader.processData()
#print(data)
prediction = MLP.predict(data)
print(prediction)
for i in range(len(prediction)):
    if prediction[i]<0.5:
        prediction[i] = 0
    else: prediction[i] = 1
from xlsReader import kaggleFileGenerator
kaggleFileGenerator('id,survived').output(data=prediction)