from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.monte_carlo import MonteCarlo
from edward.models import Normal, RandomVariable, Empirical
from edward.util import copy


class SGHMC(MonteCarlo):
  # TODO: update documentation
  """Stochastic gradient Langevin dynamics (Welling and Teh, 2011).

  Notes
  -----
  In conditional inference, we infer :math:`z` in :math:`p(z, \\beta
  \mid x)` while fixing inference over :math:`\\beta` using another
  distribution :math:`q(\\beta)`.
  ``SGLD`` substitutes the model's log marginal density

  .. math::

    \log p(x, z) = \log \mathbb{E}_{q(\\beta)} [ p(x, z, \\beta) ]
                \\approx \log p(x, z, \\beta^*)

  leveraging a single Monte Carlo sample, where :math:`\\beta^* \sim
  q(\\beta)`. This is unbiased (and therefore asymptotically exact as a
  pseudo-marginal method) if :math:`q(\\beta) = p(\\beta \mid x)`.
  """
  def __init__(self, *args, **kwargs):
    """
    Examples
    --------
    >>> z = Normal(mu=0.0, sigma=1.0)
    >>> x = Normal(mu=tf.ones(10) * z, sigma=1.0)
    >>>
    >>> qz = Empirical(tf.Variable(tf.zeros([500])))
    >>> data = {x: np.array([0.0] * 10, dtype=np.float32)}
    >>> inference = ed.SGHMC({z: qz}, data)
    """
    super(SGHMC, self).__init__(*args, **kwargs)

  def initialize(self, step_size=0.25, *args, **kwargs):
    """
    Parameters
    ----------
    step_size : float, optional
      Constant scale factor of learning rate.
    """
    print(step_size)
    self.step_size = step_size
    self.r = {z: Empirical(params=tf.Variable(tf.zeros(qz.params.get_shape())))
              for z, qz in six.iteritems(self.latent_vars)}
    return super(SGHMC, self).initialize(*args, **kwargs)

  def build_update(self):
    """
    Simulate Langevin dynamics using a discretized integrator. Its
    discretization error goes to zero as the learning rate decreases.
    """
    print("Building update.")
    old_sample = {z: tf.gather(qz.params, tf.maximum(self.t - 1, 0))
                  for z, qz in six.iteritems(self.latent_vars)}
    old_r_sample = {z: r for z, r in six.iteritems(self.r)}
    # Simulate Langevin dynamics.
    friction = tf.constant(0.1, dtype = tf.float32)
    learning_rate = tf.constant(self.step_size * 0.01, dtype = tf.float32)   # No adaptive.
    grad_log_joint = tf.gradients(self._log_joint(old_sample),
                                  list(six.itervalues(old_sample)))

    print(learning_rate)
    r_sample = {}
    # for z, qz in six.iteritems(self.latent_vars):
    #   event_shape = qz.get_event_shape()
    #   normal = Normal(mu=tf.zeros(event_shape), sigma=tf.ones(event_shape))
    #   r_sample[z] = normal.sample()

    sample = {}
    for z, qz, grad_log_p in \
        zip(six.iterkeys(self.latent_vars),
            six.itervalues(self.latent_vars),
            grad_log_joint):
      event_shape = qz.get_event_shape()
      normal = Normal(mu=tf.zeros(event_shape),
                      sigma=tf.sqrt(learning_rate * friction) * tf.ones(event_shape))
      r_sample[z] = (1. - 0.5 * friction)*old_r_sample[z] \
        + learning_rate * grad_log_p + normal.sample()
      sample[z] = old_sample[z] + r_sample[z]   # This implements eq. 15 from paper.
      # sample[z] = r_sample[z]

    # Update Empirical random variables.
    assign_ops = []
    variables = {x.name: x for x in
                 tf.get_default_graph().get_collection(tf.GraphKeys.VARIABLES)}
    for z, qz in six.iteritems(self.latent_vars):
      variable = variables[qz.params.op.inputs[0].op.inputs[0].name]
      assign_ops.append(tf.scatter_update(variable, self.t, sample[z]))
      self.r[z] = r_sample[z]

    # Increment n_accept.
    assign_ops.append(self.n_accept.assign_add(1))
    return tf.group(*assign_ops)

  def _log_joint(self, z_sample):
    """
    Utility function to calculate model's log joint density,
    log p(x, z), for inputs z (and fixed data x).

    Parameters
    ----------
    z_sample : dict
      Latent variable keys to samples.
    """
    if self.model_wrapper is None:
      scope = 'inference_' + str(id(self))
      # Form dictionary in order to replace conditioning on prior or
      # observed variable with conditioning on a specific value.
      dict_swap = z_sample.copy()
      for x, qx in six.iteritems(self.data):
        if isinstance(x, RandomVariable):
          if isinstance(qx, RandomVariable):
            qx_copy = copy(qx, scope=scope)
            dict_swap[x] = qx_copy.value()
          else:
            dict_swap[x] = qx

      log_joint = 0.0
      for z in six.iterkeys(self.latent_vars):
        z_copy = copy(z, dict_swap, scope=scope)
        z_log_prob = tf.reduce_sum(z_copy.log_prob(dict_swap[z]))
        if z in self.scale:
          z_log_prob *= self.scale[z]

        log_joint += z_log_prob

      for x in six.iterkeys(self.data):
        if isinstance(x, RandomVariable):
          x_copy = copy(x, dict_swap, scope=scope)
          x_log_prob = tf.reduce_sum(x_copy.log_prob(dict_swap[x]))
          if x in self.scale:
            x_log_prob *= self.scale[x]

          log_joint += x_log_prob
    else:
      x = self.data
      log_joint = self.model_wrapper.log_prob(x, z_sample)

    return log_joint
