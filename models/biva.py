import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from layers import (conv2d, dense, gated_resnet, transposed_gated_resnet,
                    extend_sample_dimension, stochastic_gaussian, kl_divergences,
                    variational_inference, bernoulli_log_likelihood,
                    discretized_mix_logistic_loss, sample_from_discretized_mix_logistic, nin)
from ._base_vae import _BaseVAE


class BIVA(_BaseVAE):

    def _stochastic(self, x, dim, scope, ema):
        b_init_var = tf.constant_initializer(0. if self.is_log_var else 1.)
        x = self.activation(x)
        if isinstance(dim, int):
            flatten = tf.contrib.layers.flatten(x)
            mean = dense(flatten, dim, scope=scope + "_mean", training=self.ph_is_training,
                         ema=ema, init=self.init)
            var = dense(flatten, dim, scope=scope + "_var", bias_initializer=b_init_var,
                        training=self.ph_is_training, ema=ema, init=self.init)
        else:
            mean = conv2d(x, dim, scope=scope + "_mean",
                          training=self.ph_is_training, ema=ema, init=self.init)
            var = conv2d(x, dim, scope=scope + "_var",
                         bias_initializer=b_init_var, training=self.ph_is_training, ema=ema, init=self.init)
        var = tf.nn.softplus(var) + self.eps
        z = stochastic_gaussian(mean, var, is_log_var=self.is_log_var)
        return z, mean, var

    def _inference_bu(self, x, ema):

        kwargs = {"training": self.ph_is_training, "ema": ema, "init": self.init}

        stochastic_inference_bottom_up = []
        deterministic_path_top_down = []
        deterministic_path_bottom_up = []

        skip = None
        d_bu = x
        d_td = x

        dim_length = len(self.deterministic_layers[0])
        for i, dims in enumerate(self.deterministic_layers):
            assert len(dims) == dim_length
            # Build deterministic block
            for j, dim in enumerate(dims):
                scope = "deterministic_bottom_up_%i_%i" % (i, j)
                residual = False if i == 0 and j == 0 else True

                if i > 0:
                    skip = tf.concat([deterministic_path_bottom_up[i - 1], deterministic_path_top_down[i - 1]], axis=-1)

                d_bu = gated_resnet(d_bu, skip, dim, self.activation, scope, residual, self.dropout_inference, **kwargs)

            deterministic_path_bottom_up += [d_bu]

            for j, dim in enumerate(dims):
                scope = "deterministic_top_down_%i_%i" % (i, j)
                residual = False if i == 0 and j == 0 else True
                d_td = gated_resnet(d_td, deterministic_path_bottom_up[i], dim, self.activation, scope, residual,
                                    self.dropout_inference, **kwargs)

            deterministic_path_top_down += [d_td]

            if i == len(self.deterministic_layers) - 1:
                break  # Do not add the top z layer before building the top-down inference model.

            # Build stochastic layer
            dim = self.stochastic_layers[i]
            scope = "qz_bottom_up_%i" % (i + 1)
            q_z_bottom_up, q_mean_bottom_up, q_var_bottom_up = self._stochastic(d_bu, dim, scope, ema)

            stochastic_inference_bottom_up += [(q_z_bottom_up, q_mean_bottom_up, q_var_bottom_up)]

            if len(q_z_bottom_up.get_shape()) == 2:
                flatten_shape = [int(dim) for dim in d_bu.get_shape()[1:]]
                scope = "dense2conv_bottom_up_%i" % (i + 1)
                d_bu = dense(q_z_bottom_up, np.prod(flatten_shape), scope, **kwargs)
                d_bu = tf.reshape(d_bu, [-1] + flatten_shape)
            else:
                d_bu = q_z_bottom_up

        # Build top stochastic layer q(z_L | x) from the input of bottom-up and top-down inference.
        dim = self.deterministic_layers[-1][-1]
        d = tf.concat([d_td, d_bu], axis=-1)
        scope = "deterministic_top"
        d = gated_resnet(d, None, [dim[0], dim[1], [1, 1]], self.activation, scope, True, self.dropout_inference, **kwargs)
        flatten_shapes = [[int(dim) for dim in d.get_shape()[1:]]]

        scope = "qz_top_%i" % len(self.stochastic_layers)
        q_z_top, q_mean_top, q_var_top = self._stochastic(d, self.stochastic_layers[-1], scope, ema)

        stochastic_inference_bottom_up += [(q_z_top, q_mean_top, q_var_top)]

        return stochastic_inference_bottom_up, deterministic_path_top_down, flatten_shapes

    def _inference_td(self, stochastic_inference_bottom_up, deterministic_path_top_down, flatten_shapes, ema):
        kwargs = {"training": self.ph_is_training, "ema": ema, "init": self.init}

        stochastic_inference_top_down = []

        z_top = stochastic_inference_bottom_up[-1][0]

        stochastic_inference_bottom_up = stochastic_inference_bottom_up[:-1]
        stochastic_layers_reordered = self.stochastic_layers[::-1][1:]
        deterministic_path_top_down_reordered = deterministic_path_top_down[::-1][1:]
        generative_skip_connections = []

        for i, dims in enumerate(self.deterministic_layers[::-1][:-1]):

            z_top_reshaped = z_top
            scope_index = len(self.stochastic_layers) - (i + 1)

            if len(z_top_reshaped.get_shape()) == 2:
                scope = "dense2conv_%i" % scope_index
                z_top_reshaped = dense(z_top_reshaped, np.prod(flatten_shapes[i]), scope, **kwargs)
                z_top_reshaped = tf.reshape(z_top_reshaped, [-1] + flatten_shapes[i])

            # Build deterministic block of generative model.
            d_p = z_top_reshaped
            for j, dim in enumerate(dims[::-1]):
                skip = None
                if i > 0:
                    skip = generative_skip_connections.pop()
                residual = False if j == 0 else True
                scope = "deterministic_generative_%i_%i" % (i, j)
                d_p = transposed_gated_resnet(d_p, skip, dim, self.activation, scope, residual, self.dropout_generative, **kwargs)
                generative_skip_connections = [d_p] + generative_skip_connections

            q_z_bottom_up = stochastic_inference_bottom_up[::-1][i][0]

            # Build top-down stochastic layer q(z_(L-(i+1)) | z_(L-i)) and p(z_(L-(i+1)) | z_(L-i))
            d_td = deterministic_path_top_down_reordered[i]
            scope = "qz_top_down_pz_merge_%i" % scope_index
            dim = dims[::-1][-1]
            dim = [dim[0], dim[1], [1, 1]]
            d_td = gated_resnet(d_td, d_p, dim, self.activation, scope, True, self.dropout_generative, **kwargs)

            flatten_shapes += [[int(d) for d in d_td.get_shape()[1:]]]
            scope = "qz_top_down_%i" % scope_index
            q_z_top_down, q_mean_top_down, q_var_top_down = self._stochastic(d_td, stochastic_layers_reordered[i],
                                                                             scope, ema)

            stochastic_inference_top_down += [(q_z_top_down, q_mean_top_down, q_var_top_down)]

            z_top = tf.concat([q_z_top_down, q_z_bottom_up], axis=-1)

        return stochastic_inference_top_down[::-1]

    def _generative_bu_td(self, stochastic_inference_bottom_up, stochastic_inference_top_down, flatten_shapes, ema,
                          generative_path=False):
        kwargs = {"training": self.ph_is_training, "ema": ema, "init": self.init}

        stochastic_generative_top_down = []
        stochastic_generative_bottom_up = []

        z_top = stochastic_inference_bottom_up[-1][0]

        # Build top-down inference and generative model.
        stochastic_inference_bottom_up = stochastic_inference_bottom_up[:-1]
        stochastic_layers_reordered = self.stochastic_layers[::-1][1:]
        generative_skip_connections = []

        for i, dims in enumerate(self.deterministic_layers[::-1]):

            z_top_reshaped = z_top
            scope_index = len(self.stochastic_layers) - (i + 1)

            if len(z_top_reshaped.get_shape()) == 2:
                scope = "dense2conv_%i" % scope_index
                z_top_reshaped = dense(z_top_reshaped, np.prod(flatten_shapes[i]), scope, **kwargs)
                z_top_reshaped = tf.reshape(z_top_reshaped, [-1] + flatten_shapes[i])

            # Build deterministic block of generative model.
            d_p = z_top_reshaped
            for j, dim in enumerate(dims[::-1]):
                skip = None
                if i > 0:
                    skip = generative_skip_connections.pop()
                residual = False if j == 0 else True
                scope = "deterministic_generative_%i_%i" % (i, j)
                d_p = transposed_gated_resnet(d_p, skip, dim, self.activation, scope, residual, self.dropout_generative, **kwargs)
                generative_skip_connections = [d_p] + generative_skip_connections

            # If this is the last decoding step there should be no stochastic layer.
            if i == len(self.deterministic_layers) - 1: break

            # Build bottom-up stochastic layer p(z_(L-(i+1)) | z_(L-i)).
            scope = "pz_bottom_up_%i" % scope_index
            p_z_bottom_up, p_mean_bottom_up, p_var_bottom_up = self._stochastic(d_p, stochastic_layers_reordered[i],
                                                                                scope, ema)

            stochastic_generative_bottom_up += [(p_z_bottom_up, p_mean_bottom_up, p_var_bottom_up)]

            q_z_bottom_up = stochastic_inference_bottom_up[::-1][i][0]

            # Build top-down stochastic layer q(z_(L-(i+1)) | z_(L-i)) and p(z_(L-(i+1)) | z_(L-i))
            q_z_top_down = stochastic_inference_top_down[::-1][i][0]

            scope = "pz_top_down_%i" % scope_index
            p_z_top_down, p_mean_top_down, p_var_top_down = self._stochastic(d_p, stochastic_layers_reordered[i], scope,
                                                                             ema)

            stochastic_generative_top_down += [(p_z_top_down, p_mean_top_down, p_var_top_down)]

            if generative_path:
                z_top = tf.concat([p_z_top_down, p_z_bottom_up], axis=-1)
            else:
                z_top = tf.concat([q_z_top_down, q_z_bottom_up], axis=-1)

        return d_p, stochastic_generative_bottom_up[::-1], stochastic_generative_top_down[::-1]

    def _generate_output_nll(self, d_p, ema, x=None):
        kwargs = {"training": self.ph_is_training, "ema": ema, "init": self.init}

        log_likelihood = None
        if self.output_distribution == self.distributions[0]:  # Bernoulli output.
            x_out = nin(self.activation(d_p), self.input_shape[-1], scope="dense", **kwargs)

            op_output = tf.sigmoid(x_out)

            # Sample into binary values.
            shp = tf.shape(op_output)
            op_output_sample = tf.where(tf.random_uniform(shp) - op_output < 0, tf.ones(shp), tf.zeros(shp))

            if x is not None:
                shape_flatten = [-1] + [np.prod(self.input_shape)]
                log_likelihood = bernoulli_log_likelihood(tf.reshape(x, shape_flatten),
                                                          tf.reshape(x_out, shape_flatten))

        elif self.output_distribution == self.distributions[2]:  # Discretized mixture likelihood output.
            nr_mix = 10
            x_out = nin(self.activation(d_p), nr_mix * 10, scope="dense", **kwargs)

            levels = 256.
            op_output = tf.div(
                (sample_from_discretized_mix_logistic(x_out, nr_mix) * (levels - 1) / 2.) + (levels - 1) / 2.,
                ((levels - 1)))  # Sample and reshape
            op_output_sample = op_output

            if x is not None:
                log_likelihood = discretized_mix_logistic_loss(x, x_out, levels)

        return op_output, op_output_sample, log_likelihood

    def _construct(self, ema):

        with tf.variable_scope(self.__class__.__name__, reuse=tf.AUTO_REUSE):
            x_extended = extend_sample_dimension(self.ph_input, self.ph_eq, self.ph_iw, name="sample_layer")

            if self.output_distribution == self.distributions[2]:
                x_extended = (2. * x_extended) - 1.  # Rescale between -1 and 1

            # For data-dependent initialization
            self.init = True
            stochastic_inference_bottom_up, deterministic_path_top_down, flatten_shapes = self._inference_bu(x_extended,
                                                                                                             ema)
            stochastic_inference_top_down = self._inference_td(stochastic_inference_bottom_up,
                                                               deterministic_path_top_down, flatten_shapes, ema)
            op_data_dependent_initialization1 = tf.group(stochastic_inference_bottom_up, stochastic_inference_top_down)
            d_p, stochastic_generative_bottom_up, stochastic_generative_top_down = self._generative_bu_td(
                stochastic_inference_bottom_up, stochastic_inference_top_down, flatten_shapes, ema,
                generative_path=False)
            op_data_dependent_initialization2, _, _ = self._generate_output_nll(d_p, ema, x_extended)
            self.op_data_dependent_initialization = tf.group(op_data_dependent_initialization1,
                                                             op_data_dependent_initialization2)
            self.init = False

            # Build bottom-up (bu) inference model and top-down (td) deterministic path.
            stochastic_inference_bottom_up, deterministic_path_top_down, flatten_shapes = self._inference_bu(x_extended,
                                                                                                             ema)
            stochastic_inference_top_down = self._inference_td(stochastic_inference_bottom_up,
                                                               deterministic_path_top_down, flatten_shapes, ema)
            d_p, stochastic_generative_bottom_up, stochastic_generative_top_down = self._generative_bu_td(
                stochastic_inference_bottom_up, stochastic_inference_top_down, flatten_shapes, ema,
                generative_path=False)
            self.op_output, self.op_output_sample, ll = self._generate_output_nll(d_p, ema, x_extended)

            # Collect the <q,p> layers in lists
            assert len(stochastic_inference_bottom_up) - 1 == len(stochastic_inference_top_down) == len(
                stochastic_generative_top_down) == len(stochastic_generative_bottom_up)

            def stochastic_flatten(z, mean, var):
                return flatten(z), flatten(mean), flatten(var)

            q_layers = []
            p_layers = []
            self.q_layers_spatial = []
            self.p_layers_spatial = []

            for q_bu, q_td, p_bu, p_td in zip(stochastic_inference_bottom_up[:-1], stochastic_inference_top_down,
                                              stochastic_generative_bottom_up, stochastic_generative_top_down):
                q_layers += [stochastic_flatten(*q_bu)]
                self.q_layers_spatial += [q_bu]
                p_layers += [stochastic_flatten(*p_bu)]
                self.p_layers_spatial += [p_bu]

                q_layers += [stochastic_flatten(*q_td)]
                self.q_layers_spatial += [q_td]
                p_layers += [stochastic_flatten(*p_td)]
                self.p_layers_spatial += [p_td]

            q_layers += [stochastic_flatten(*stochastic_inference_bottom_up[-1])]
            self.q_layers_spatial += [stochastic_inference_bottom_up[-1]]

            self.op_kls = kl_divergences(q_layers, p_layers, self.is_log_var)

            self.op_loss = tf.reduce_mean(
                -variational_inference(ll, self.op_kls, self.ph_eq, self.ph_iw, self.minimum_kl, self.ph_temp,
                                       self.ph_is_training))
            self.op_nll = tf.reduce_mean(-ll)

            # For evaluation
            d_p, _, _ = self._generative_bu_td(stochastic_inference_bottom_up, stochastic_inference_top_down,
                                               flatten_shapes, ema, generative_path=True)
            _, self.op_generate, _ = self._generate_output_nll(d_p, ema)
            self.op_inference = self._inference_bu(x_extended, ema)[0][-1][0]

    def _define_model(self):
        self._construct(None)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__class__.__name__)

        if self.ema:
            self.maintain_averages_op = tf.group(self.ema.apply(self.trainable_variables))
            self._construct(self.ema)

        self.global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.__class__.__name__)
