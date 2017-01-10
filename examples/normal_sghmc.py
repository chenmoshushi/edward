#!/usr/bin/env python
"""Correlated normal posterior. Inference with stochastic gradient
Langevin dynamics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from edward.models import Empirical, MultivariateNormalFull

ed.set_seed(42)

# MODEL
z = MultivariateNormalFull(
    mu=tf.ones(2),
    sigma=tf.constant([[1.0, 0.8], [0.8, 1.0]]))

# INFERENCE
n_samples = 10
qz = Empirical(params=tf.Variable(tf.random_normal([n_samples, 2])))

# inference = ed.HMC({z: qz})
# inference.run(step_size=1.)

# inference = ed.SGLD({z: qz})
# inference.run(step_size = 25.)

# CRITICISM
# sess = ed.get_session()
# mean, std = sess.run([qz.mean(), qz.std()])
# print("Inferred posterior mean:")
# print(mean)
# print("Inferred posterior std:")
# print(std)


def make_contour_plot(to_label = True):
    # Sample from true
    delta = 0.025
    x = np.arange(-3, 3, delta)
    y = np.arange(-3, 3, delta)
    X, Y = np.meshgrid(x, y)
    T = tf.convert_to_tensor(np.c_[X.flatten(), Y.flatten()], dtype=tf.float32)
    D = sess.run(tf.exp(z.log_prob(T)))
    Z = D.reshape((len(x), len(x)))
    cs = plt.contour(X, Y, Z)
    if to_label:
        plt.clabel(cs, inline=1, fontsize=10)

def demo_hmc():
    "HMC demo with parameters tuned to make things look reasonable."
    n_samples = 10000
    qz = Empirical(params=tf.Variable(tf.random_normal([n_samples, 2])))
    inference = ed.HMC({z: qz})
    inference.run(step_size = 0.3)
    var = qz.get_variables()[0]
    val = var.value()
    trace = val.eval()
    fig, ax = plt.subplots(1)
    ax.scatter(trace[::100,0], trace[::100,1], marker = ".")
    plt.hold(True)
    make_contour_plot()
    plt.show()

def demo_sgld():
    "SGLD demo with parameters tuned to make things look reasonable."
    n_samples = 50000
    qz = Empirical(params=tf.Variable(tf.random_normal([n_samples, 2])))
    inference = ed.SGLD({z: qz})
    inference.run(step_size = 25.)
    var = qz.get_variables()[0]
    val = var.value()
    trace = val.eval()
    fig, ax = plt.subplots(1)
    ax.scatter(trace[::500,0], trace[::500,1], marker = ".")
    plt.hold(True)
    make_contour_plot()
    plt.show()

def demo_bnn():
    """
    Demo for Tianqi's neural net.
    """
    # Single-hidden-layer Bayesian nnet with 100 hidden nodes; sigmoid hidden,
    # softmax output.
    pass

if __name__ == '__main__':
    # Run SGHMC
    n_samples = 5000
    #qz = Empirical(params=tf.Variable(tf.random_normal([n_samples, 2])))
    qz = Empirical(params=tf.Variable(tf.zeros([n_samples, 2])))
    inference = ed.SGHMC({z: qz})
    inference.run(step_size=1.)
    trace = qz.get_variables()[0].value().eval()
    r_trace = inference.r.values()[0].value().eval()
    fig, ax = plt.subplots(1)
    ax.scatter(trace[::50,0], trace[::50,1], marker = ".")
    #ax.plot(trace[::50,0], trace[::50,1], marker = ".")
    plt.hold(True)
    make_contour_plot()
    plt.show()
