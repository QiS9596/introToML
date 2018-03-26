import numpy as np
import tensorflow as tf
import re

def loadDataIntoNDArray(filename:str):
    """
    this method loads file with filename as it's name and returns ndarray object contains the data inside the file
    :param filename:path to target file
    :return: ndarray
    """
    with open(filename) as file:
        temp = file.readlines()
        for i in range(0,len(temp)):
            temp[i] = re.split(' ',temp[i].replace('\n',''))
            for ii in range(0,len(temp[i])):
                temp[i][ii] = int(temp[i][ii])
        temp = np.array(temp)
        return temp

prefix = './HW1-1/'
data = 'data.txt'
answer = 'answer.txt'

train_1 = loadDataIntoNDArray(prefix+data)
train_2 = loadDataIntoNDArray(prefix+answer).transpose()

# train_1 = np.array([[1., 2., 3.], [3., 4., 5.], [8., 5., 7.], [7., 2., 8.]])
#train_2 = np.array([[1.], [0.], [0.], [1.]])

input_1 = tf.placeholder(tf.float32, shape=[None, 32])
input_2 = tf.placeholder(tf.float32, shape=[None, 1])

"""
3 dimensional input
2 hidden input layer node -> 2 dimensional output for input layer, 2 input for output layer
1 dimensional output for output layer(layer2)
"""
# defining input layer
weight_1 = tf.get_variable(name='weight_1', shape=[32, 2], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_1 = tf.get_variable(name='bias_1', shape=[2], dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_1_output = tf.add(tf.matmul(input_1, weight_1), bias_1)

# output layer
weight_2 = tf.get_variable(name='weight_2', shape=[2, 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
bias_2 = tf.get_variable(name="bias_2", shape=[1], dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
layer_2_output = tf.sigmoid(tf.add(tf.matmul(layer_1_output, weight_2), bias_2))

loss = tf.losses.mean_squared_error(train_2, layer_2_output)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(0, 1000):
        if i % 20 == 0:
            print('loss : ', sess.run(loss, feed_dict={input_1: train_1, input_2: train_2}))
            print('predict : ', sess.run(layer_2_output, feed_dict={input_1: train_1, input_2: train_2}))
        sess.run(train, feed_dict={input_1: train_1, input_2: train_2})
        if sess.run(loss, feed_dict={input_1: train_1, input_2: train_2}) < 0.0003:
            break
