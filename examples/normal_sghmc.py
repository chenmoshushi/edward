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

from edward.models import Empirical, MultivariateNormalFull

ed.set_seed(42)

# MODEL
z = MultivariateNormalFull(
    mu=tf.ones(2),
    sigma=tf.constant([[1.0, 0.8], [0.8, 1.0]]))

# INFERENCE
n_samples = 2000
qz = Empirical(params=tf.Variable(tf.random_normal([n_samples, 2])))

inference = ed.SGHMC({z: qz})
# inference = ed.HMC({z: qz})
inference.run(step_size=.1)
# inference = ed.SGLD({z: qz})
# inference.run(step_size = 3.)

# CRITICISM
sess = ed.get_session()
mean, std = sess.run([qz.mean(), qz.std()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior std:")
print(std)

# Retrieve sampling traces.
var = qz.get_variables()[0]
val = var.value()
trace = val.eval()
# fig, ax = plt.subplots(2)
# ax[0].plot(trace[:,0])
# ax[1].plot(trace[:,1])
# plt.show()
fig, ax = plt.subplots(1)
ax.scatter(trace[:,0], trace[:,1])
plt.show()
