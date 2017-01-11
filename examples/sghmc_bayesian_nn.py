
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import six
import numpy as np
import tensorflow as tf

import edward as ed

from edward.models import Normal, Categorical, Multinomial, Empirical


def neural_network(x):
    h = tf.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_2) + b_2
    return h


if __name__ == "__main__":

    tf.reset_default_graph()

    ed.set_seed(42)

    # load MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    # MODEL
    nhidden = 100
    W_0 = Normal(mu=tf.zeros([784, nhidden]), sigma=tf.ones([784, nhidden]))
    W_1 = Normal(mu=tf.zeros([nhidden, nhidden]), sigma=tf.ones([nhidden, nhidden]))
    W_2 = Normal(mu=tf.zeros([nhidden, 10]), sigma=tf.ones([nhidden, 10]))
    b_0 = Normal(mu=tf.zeros(nhidden), sigma=tf.ones(nhidden))
    b_1 = Normal(mu=tf.zeros(nhidden), sigma=tf.ones(nhidden))
    b_2 = Normal(mu=tf.zeros(10), sigma=tf.ones(10))
    
    batch_size = 500
    x_ph = tf.placeholder(tf.float32, [batch_size, 784])
    y_ph = tf.placeholder(tf.float32, [batch_size])

    # Really want to use Multinomial for one-hot
    y = Categorical(logits=neural_network(x_ph))

    nsamples = 5000
    qW_0 = Empirical(params=tf.Variable(tf.zeros([nsamples, 784, nhidden])))
    qW_1 = Empirical(params=tf.Variable(tf.zeros([nsamples, nhidden, nhidden])))
    qW_2 = Empirical(params=tf.Variable(tf.zeros([nsamples, nhidden, 10])))
    qb_0 = Empirical(params=tf.Variable(tf.zeros([nsamples, nhidden])))
    qb_1 = Empirical(params=tf.Variable(tf.zeros([nsamples, nhidden])))
    qb_2 = Empirical(params=tf.Variable(tf.zeros([nsamples, 10])))

    inference = ed.SGHMC({W_0: qW_0, b_0: qb_0,
                          W_1: qW_1, b_1: qb_1,
                          W_2: qW_2, b_2: qb_2}, data={y: y_ph})
    # may need `scale` parameter...
    inference.initialize()

    tf.initialize_all_variables()

    for _ in xrange(nsamples):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_ys_not_hot = tf.argmax(batch_ys, 1)
        inference.update(feed_dict={x_ph: batch_xs, y_ph: batch_ys_not_hot})
