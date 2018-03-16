from time import sleep

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

x = tf.placeholder(tf.float32, shape=(1, 10))
y = tf.placeholder(tf.float32, shape=(1, 1))

y_ = 2 * slim.fully_connected(inputs=x, num_outputs=1, activation_fn=None)

loss = tf.reduce_mean(tf.square(tf.subtract(y, y_)))
learning_rate = 0.001

tvs = tf.trainable_variables()
w1 = tvs[0]

new_w = tf.placeholder(tf.float32, shape=(10, 1))
assign_op = tf.assign(w1, new_w)

grads = tf.gradients(loss, w1)[0]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

x_input = np.ones(shape=(1, 10,))
y_input = np.ones(shape=(1, 1))

print(tvs)
feed_dict = {x: x_input, y: y_input}

for i in range(1000):
    print(sess.run([y_, y, tvs], feed_dict))

    grad_values = sess.run(grads, feed_dict)

    cur_w_values = sess.run(w1, feed_dict)
    new_w_values = cur_w_values - learning_rate * np.expand_dims(grad_values.flatten(), axis=1)
    sess.run(assign_op, {new_w: new_w_values})

    print('loss = {}'.format(sess.run(loss, feed_dict)))
    sleep(.5)
