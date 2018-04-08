"""
Autoencoder

Ref:Tensorflow in action chapter 4
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant=1):
    """
    function to initialize weight. The weights initialized by xavier_init should have 2/(fan_in+fan_out) as it's variance
    :param fan_in: number of input node
    :param fan_out: number of output node
    :param constant:
    :return:
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        """

        :param n_input: input number
        :param n_hidden: hidden layer node number
        :param transfer_function: activation function, use softplus as default
        :param optimizer: use adam as default
        :param scale: scale of gaussian noise
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)  # ?
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # defination of the architecture of the neural network
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']
        ))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # mean square error
        # self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.cost = tf.losses.mean_squared_error(self.reconstruction, self.x)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        """
        initialize weights
        :return: a dictionary contains weights and biases
        """
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        """
        using one batch as training set and returns the current cost
        :param X: one of the mini batches
        :return: current cost
        """
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        """
        calculate one cost for current nod on the graph
        :param X:
        :return:
        """
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        """
        get the output of the hidden layer, the purpose is to learn the abstract characteristics of the data
        :param X: input data
        :return: hidden layer output
        """
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        """
        using hidden layer output as it's input, reconstruct the origin data using the abstract characteristics
        :param hidden: hidden layer output
        :return: decoded information
        """
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        """
        do transform and generate process
        :param X: origin data
        :return: reconstructed data
        """
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):  # why do so?
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


def initializeMainFunction():
    global mnist
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def standard_scale(X_train, X_test):
    """
    normalize data, fitting it into normalized form with E = 0, Var = 1,
    note: we should let the scaler fit the training set and use the same set to handle the testing set
    :param X_train: training set
    :param X_test: testing set
    :return: normalized training set and testing set
    """
    preprossor = StandardScaler().fit(X_train)
    X_train = preprossor.transform(X_train)
    X_test = preprossor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    """
    select block size data, select batch
    :param data: data full set
    :param batch_size: the size of a single batch
    :return: random selected batch
    """
    start_idx = np.random.randint(0, len(data) - batch_size)
    return data[start_idx:(start_idx + batch_size)]


global X_train, autoencorder


def train():
    initializeMainFunction()
    # normalization of training and testing dataset
    global X_train, autoencorder
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    autoencorder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus,
                                                    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
    # start training
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)

            cost = autoencorder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
            print("Epoch: ", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    print("Total cost: " + str(autoencorder.calc_total_cost(X_test)))


train()

print(X_train[0])
encoded = autoencorder.transform([X_train[0]])
print(encoded)
decoded = autoencorder.generate(hidden=encoded)
print(decoded)
