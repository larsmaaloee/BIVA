# External
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base


class _ExtendSampleDimension(base.Layer):
    def __init__(self, eq, iw, name=None, **kwargs):
        super(_ExtendSampleDimension, self).__init__(trainable=False, name=name, **kwargs)
        self.eq = eq
        self.iw = iw

    def build(self, input_shape):
        if not len(input_shape) > 1:
            raise ValueError('Inputs should have at least rank 2. '
                             'Received input shape:', str(input_shape))

    def call(self, inputs, **kwargs):
        # Expand and tile the EQ and IW dimension.
        trailing_dims = [int(d) for d in inputs.get_shape()[1:]]
        trailing_multiplies = list(np.sign(trailing_dims))
        outputs = tf.tile(tf.expand_dims(tf.expand_dims(inputs, 1), 1), [1, self.eq, self.iw] + trailing_multiplies)
        outputs = tf.reshape(outputs, [-1] + trailing_dims)
        return outputs


def extend_sample_dimension(inputs, eq, iw, name=None):
    layer = _ExtendSampleDimension(eq, iw, name)
    return layer.apply(inputs)


class _StochasticGaussian(base.Layer):
    def __init__(self, is_log_var, mean=0, std=1., seed=1234, name=None, **kwargs):
        super(_StochasticGaussian, self).__init__(trainable=False, name=name, **kwargs)
        self.is_log_var = is_log_var
        self.mean = mean
        self.std = std
        self.seed = seed

    def build(self, input_shape):
        if not len(input_shape) == 2:
            raise ValueError('Inputs should have at least rank 2, mean and variance tensor. '
                             'Received input shape:', str(input_shape))

    def call(self, inputs, **kwargs):
        input_mean, input_var = inputs

        output_shape = tf.shape(input_mean)

        input_var = tf.exp(0.5 * input_var) if self.is_log_var else tf.sqrt(input_var)

        eps = tf.reshape(tf.random_normal(output_shape, mean=self.mean, stddev=self.std, seed=self.seed, name=self.name), output_shape)
        outputs = input_mean + input_var * eps

        return outputs


def stochastic_gaussian(inputs_mean, inputs_var, is_log_var, mean=0., std=1., seed=1234, name=None):
    layer = _StochasticGaussian(is_log_var, mean, std, seed, name)
    return layer.apply([inputs_mean, inputs_var])


class _GaussianMerge(base.Layer):
    def __init__(self, is_log_var=False, eps=1e-8, name=None, **kwargs):
        super(_GaussianMerge, self).__init__(trainable=False, name=name, **kwargs)
        self.is_log_var = is_log_var
        self.eps = eps

    def build(self, input_shape):
        if not len(input_shape) == 4:
            raise ValueError('Inputs should have at least rank 4, mean and variance tensor for each of the two merges. '
                             'Received input shape:', str(input_shape))

    def call(self, inputs, **kwargs):
        input_mean_1, input_var_1, input_mean_2, input_var_2 = inputs

        input_var_1 = tf.exp(input_var_1) if self.is_log_var else input_var_1
        input_var_2 = tf.exp(input_var_2) if self.is_log_var else input_var_2

        input_prec_1 = tf.pow(tf.maximum(input_var_1, self.eps), -1)
        input_prec_2 = tf.pow(tf.maximum(input_var_2, self.eps), -1)

        mean_merge = (input_mean_1 * input_prec_1 + input_mean_2 * input_prec_2) / (input_prec_1 + input_prec_2)
        var_merge = tf.pow((input_prec_1 + input_prec_2), -1)

        if self.is_log_var:
            var_merge = tf.log(var_merge)

        return mean_merge, var_merge


def gaussian_merge(inputs_mean_1, inputs_var_1, inputs_mean_2, inputs_var_2, is_log_var, eps=1e-8, name=None):
    layer = _GaussianMerge(is_log_var, eps, name)
    return layer.apply([inputs_mean_1, inputs_var_1, inputs_mean_2, inputs_var_2])

