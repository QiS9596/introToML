import tensorflow as tf


a = tf.Variable(name='weight',dtype=tf.float32, initial_value=[3])
print(a.name)
b = tf.Variable(name='weight',dtype=tf.float32, initial_value = [4])
print(b.name)
print(a.name)