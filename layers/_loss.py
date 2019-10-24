import math

import tensorflow as tf
import numpy as np
from tensorflow.python.layers import base


class _VariationalInference(base.Layer):
    def __init__(self, eq, iw, minimum_kl=0., temperature=1., training=True, name=None, **kwargs):
        super(_VariationalInference, self).__init__(trainable=False, name=name, **kwargs)
        self.eq = eq
        self.iw = iw
        self.temperature = temperature
        self.minimum_kl = minimum_kl
        self.is_training = training

    def build(self, input_shape):
        if not len(input_shape) == 4:
            raise ValueError('Inputs should have at least rank 4. '
                             'Received input shape:', str(input_shape))

    def reshape_sample_dims(self, x):
        return tf.reshape(x, [-1, self.eq, self.iw, 1])

    def __call__(self, inputs, **kwargs):
        ll, kls = inputs

        kl_sum = None

        for i, kl in enumerate(kls):
            kl = tf.cond(self.is_training, lambda: tf.minimum(self.minimum_kl, kl), lambda: kl)
            kl_sum = kl if kl_sum is None else kl_sum + kl

        kl = self.reshape_sample_dims(kl_sum)
        ll = self.reshape_sample_dims(ll)
        return tf.reduce_mean(log_sum_exp(ll + self.temperature * kl, axis=2, sum_op=tf.reduce_mean), axis=1)


def variational_inference(log_likelihood, kl_divergences, eq, iw, minimum_kl, temperature, training, name=None):
    layer = _VariationalInference(eq, iw, minimum_kl, temperature, training, name)
    return layer.apply([log_likelihood, kl_divergences])


class _GaussianLogLikelihood(base.Layer):
    def __init__(self, is_log_var, eps, name=None, **kwargs):
        super(_GaussianLogLikelihood, self).__init__(name=name, **kwargs)
        self.is_log_var = is_log_var
        self.eps = eps

    def __call__(self, inputs, **kwargs):
        input, output_mean, output_var = inputs
        log_px = log_normal(input, output_mean, output_var, is_log_var=self.is_log_var, eps=self.eps)
        return log_px


def gaussian_log_likelihood(input, output_mean, output_var, is_log_var, eps=1e-8, name=None):
    layer = _GaussianLogLikelihood(is_log_var, eps, name)
    return layer.apply([input, output_mean, output_var])


class _BernoulliLogLikelihood(base.Layer):
    def __init__(self, name=None, **kwargs):
        super(_BernoulliLogLikelihood, self).__init__(name=name, **kwargs)

    def __call__(self, inputs, **kwargs):
        input, output_mean = inputs
        log_px = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(labels=input, logits=output_mean), axis=-1, keepdims=True)
        return log_px

def bernoulli_log_likelihood(input, output, name=None):
    layer = _BernoulliLogLikelihood(name)
    return layer.apply([input, output])


class _CategoricalLogLikelihood(base.Layer):
    def __init__(self, name=None, **kwargs):
        super(_CategoricalLogLikelihood, self).__init__(name=name, **kwargs)

    def __call__(self, inputs, **kwargs):
        input, output_mean = inputs

        bs, h, w, c, l = input.get_shape().as_list()
        log_px = -tf.nn.softmax_cross_entropy_with_logits(labels=input, logits=output_mean)
        log_px = tf.reshape(log_px, [-1, h, w, c])

        log_px = tf.expand_dims(tf.reduce_sum(log_px, axis=(1, 2, 3)), axis=-1)
        log_px = tf.reshape(log_px, [tf.shape(input)[0], 1])
        return log_px

def categorical_log_likelihood(input, output, name=None):
    layer = _CategoricalLogLikelihood(name)
    return layer.apply([input, output])


class _KLDivergence(base.Layer):
    def __init__(self, is_log_var, name=None, **kwargs):
        super(_KLDivergence, self).__init__(trainable=False, name=name, **kwargs)

        self.is_log_var = is_log_var

    def build(self, input_shape):
        if not len(input_shape) == 2:
            raise ValueError('Inputs should have at least rank 2. '
                             'Received input shape:', str(input_shape))

    def __call__(self, inputs, **kwargs):
        q_layers, p_layers = inputs

        kls = []
        for i in range(len(q_layers)):
            x, mean, var = q_layers[i]
            var = var if self.is_log_var else tf.log(var)
            log_pdf_q = log_normal(x, mean, var, is_log_var=True)

            if not i == (len(q_layers) - 1):
                _, mean, var = p_layers[i]
                var = var if self.is_log_var else tf.log(var)
                log_pdf_p = log_normal(x, mean, var, is_log_var=True)

            else:
                log_pdf_p = log_standard_normal(x)

            kls += [log_pdf_p - log_pdf_q]

        return kls


def kl_divergences(q_layers, p_layers, is_log_var):
    layer = _KLDivergence(is_log_var)
    return layer.apply([q_layers, p_layers])


c = - 0.5 * math.log(2 * math.pi)


def log_normal(x, mean, var, is_log_var=False):
    if is_log_var:
        log_pdf = c - var / 2 - (x - mean) ** 2 / (2 * tf.exp(var))
    else:
        log_pdf = c - tf.log(var) / 2 - (x - mean) ** 2 / (2 * var)
    return tf.reduce_sum(log_pdf, axis=-1, keepdims=True)

def log_standard_normal(x):
    return tf.reduce_sum(c - x ** 2 / 2, axis=-1, keepdims=True)


def log_bernoulli(x, p, eps=0.0):
    p = tf.clip_by_value(p, eps, 1.0 - eps)
    return tf.reduce_sum(x * tf.log(p) + (1.0 - x) * tf.log(1.0 - p), axis=-1, keepdims=True)


def log_sum_exp(A, axis=None, sum_op=tf.reduce_mean):
    A_max = tf.reduce_max(A, axis=axis, keepdims=True)
    B = tf.log(sum_op(tf.exp(A - A_max), axis=axis, keepdims=True)) + A_max
    return B


def log_sum_exp2(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keepdims=True))


def discretized_mix_logistic_loss(x, l, levels, sum_all=False):
    """
    This function is copied from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py in reference to:
    See [Salimans et. al., 2017](https://arxiv.org/pdf/1701.05517)
    ([pdf](https://arxiv.org/pdf/1701.05517.pdf))

    log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    """

    xs = [-1] + x.get_shape().as_list()[1:]  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = [-1] + l.get_shape().as_list()[1:]  # predicted distribution, e.g. (B,32,32,100)

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(tf.concat([tf.shape(x), [nr_mix]], axis=0))
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                    * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                    coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    means = tf.concat([tf.reshape(means[:, :, :, 0, :], [
        xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / (levels-1))
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / (levels-1))
    cdf_min = tf.nn.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min,
                                                            tf.where(cdf_delta > 1e-5,
                                                                     tf.log(tf.maximum(cdf_delta, 1e-12)),
                                                                     log_pdf_mid - np.log((levels-1)/2.))))

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return tf.reduce_sum(log_sum_exp2(log_probs))
    else:
        return tf.reduce_sum(log_sum_exp2(log_probs), [1, 2])


