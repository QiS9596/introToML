from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
# images are 28*28 graphs, and here we simplify it as a 784*1 array, the labels classifies the images into 10 classes

import tensorflow as tf

sess = tf.InteractiveSession()  # set sess as default session, sessions are independent in tensorflow
x = tf.placeholder(tf.float32, [None, 784])  # placeholder can be used to input data, tf.float32 is the data type
# the second parameter of tf.placeholder says that there are unlimited number of input data, whereas each one is a
# 784*1 array

"""
Using Softmax Regression
initializing weight and biases
weight is a [784,10]variable, since there is 784 dimension in x and 10 classes
"""

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)  # tf.nn defines lots of tools for building neural network

# using cross-entropy as loss
y_ = tf.placeholder(tf.float32, [None, 10])  # interface for inputting labels
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# using Stochastic Gradient Descent method as Optimization Method
train_stpe = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(0,1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_stpe.run({x:batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
