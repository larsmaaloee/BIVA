
import tensorflow as tf
import warnings


def conv2d(x, dim=(32, [3, 3], [1, 1]), pad='SAME', scope="conv2d", training=True, ema=None, init=False, bias_initializer=tf.constant_initializer(0.)):
    num_filters, filter_size, stride = dim
    with tf.variable_scope(scope):
        V = tf.get_variable('V', shape=list(filter_size) + [int(x.get_shape()[-1]), num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)

        g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                              initializer=bias_initializer, trainable=True)

        def maybe_avg(v):
            if ema is not None and not init:
                v = tf.cond(training, lambda: v, lambda: ema.average(v))
            return v

        if init:
            x = tf.nn.conv2d(x, tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2]), [1] + list(stride) + [1], pad)

            init_scale=.01
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x - tf.reshape(m_init, [1, 1, 1, num_filters]))
        else:
            V = maybe_avg(V)
            g = maybe_avg(g)
            b = maybe_avg(b)

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + list(stride) + [1], pad), b)

    return x

def gated_resnet(x, aux, dim=(32, [3, 3], [1, 1]), activation=tf.nn.elu, scope="gated_resnet", residual=True, dropout=.0, conv=conv2d, training=True, ema=None, init=False):
    out = conv(activation(x), [dim[0], dim[1], [1, 1]], scope="%s_conv_in"%scope, training=training, ema=ema, init=init)
    in_shp = x.get_shape().as_list()
    assert in_shp[1] == in_shp[2]


    if aux is not None:
        aux_shp = aux.get_shape().as_list()

        assert aux_shp[1] == aux_shp[2]
        if aux_shp[1:-1] > in_shp[1:-1]:
            aux = conv(activation(aux), [dim[0], dim[1], [aux_shp[1] // in_shp[1], aux_shp[2] // in_shp[2]]],
                       scope="%s_conv_downsample_aux" % scope, training=training, ema=ema, init=init)
        elif aux_shp[1:-1] < in_shp[1:-1]:
            aux = deconv2d(activation(aux), [dim[0], dim[1], [in_shp[1] // aux_shp[1], in_shp[2] // aux_shp[2]]],
                        scope="%s_conv_upsample_aux" % scope, training=training, ema=ema, init=init)
        else:
            aux = nin(activation(aux), dim[0], training=training, ema=ema, init=init, scope="%s_conv_aux" % scope)

        out += aux

    out = activation(out)

    if dropout > 0:
        out = tf.layers.dropout(out, rate=dropout, training=training)

    out = conv(out, [2*dim[0], dim[1], dim[2]], scope="%s_conv_out"%scope, training=training, ema=ema, init=init)
    h_stack1, h_stack2 = tf.split(out, 2, 3)
    sigmoid_out = tf.sigmoid(h_stack2)
    out = (h_stack1 * sigmoid_out)

    out_shp = out.get_shape().as_list()
    if out_shp[1:-1] < in_shp[1:-1]:
        x = tf.nn.avg_pool(x, [1, dim[2][0], dim[2][1], 1], strides=[1, dim[2][0], dim[2][1], 1], padding='SAME')
    elif out_shp[1:-1] > in_shp[1:-1]:
        warnings.warn("The height and width of the output are larger than the input. There will be no residual connection.")
        residual = False

    if out_shp[-1] > in_shp[-1]:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, int(dim[0] - in_shp[-1])]])
    elif out_shp[-1] < in_shp[-1]:
        warnings.warn("The input has more feature maps than the output. There will be no residual connection.")
        residual = False

    if residual:
        out += x

    return out


def deconv2d(x, dim=(32, [3, 3], [1, 1]), pad='SAME', scope="deconv2d", training=True, ema=None, init=False, bias_initializer=tf.constant_initializer(0.)):
    num_filters, filter_size, stride = dim

    xs = x.get_shape().as_list()
    if pad=='SAME':
        target_shape = [tf.shape(x)[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [tf.shape(x)[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]

    with tf.variable_scope(scope):
        V = tf.get_variable("V", shape=list(filter_size) + [num_filters, int(x.get_shape()[-1])], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = tf.get_variable("g", shape=[num_filters], dtype=tf.float32, initializer=tf.constant_initializer(1.), trainable=True)
        b = tf.get_variable("b", shape=[num_filters], dtype=tf.float32, initializer=bias_initializer, trainable=True)

        def maybe_avg(v):
            if ema is not None and not init:
                v = tf.cond(training, lambda: v, lambda: ema.average(v))
            return v

        if init:
            x = tf.nn.conv2d_transpose(x, tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3]), target_shape, [1] + list(stride) + [1], padding=pad)

            init_scale = .01
            m_init, v_init = tf.nn.moments(x, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x - tf.reshape(m_init, [1, 1, 1, num_filters]))

        else:
            V = maybe_avg(V)
            g = maybe_avg(g)
            b = maybe_avg(b)

            W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])
            # calculate convolutional layer output
            x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + list(stride) + [1], padding=pad)
            x = tf.nn.bias_add(x, b)

    return x


def transposed_gated_resnet(x, aux, dim=(32, [3, 3], [1, 1]), activation=tf.nn.elu, scope="transposed_gated_resnet", residual=True, dropout=.0, conv=conv2d, training=True, ema=None, init=False):
    out = conv(activation(x), [dim[0], dim[1], [1, 1]], scope="%s_conv_in" % scope, training=training, ema=ema, init=init)
    in_shp = x.get_shape().as_list()
    assert in_shp[1] == in_shp[2]

    if aux is not None:
        aux_shp = aux.get_shape().as_list()

        assert aux_shp[1] == aux_shp[2]

        if aux_shp[1:-1] > in_shp[1:-1]:
            aux = conv(activation(aux), [dim[0], dim[1], [aux_shp[1] // in_shp[1], aux_shp[2] // in_shp[2]]],
                       scope="%s_conv_downsample_aux" % scope, training=training, ema=ema, init=init)
        elif aux_shp[1:-1] < in_shp[1:-1]:
            aux = deconv2d(activation(aux), [dim[0], dim[1], [in_shp[1] // aux_shp[1], in_shp[2] // aux_shp[2]]],
                           scope="%s_conv_upsample_aux" % scope, training=training, ema=ema, init=init)
        else:
            aux = nin(activation(aux), dim[0], training=training, ema=ema, init=init, scope="%s_conv_aux" % scope)

        out += aux

    out = activation(out)

    if dropout > 0:
        out = tf.layers.dropout(out, rate=dropout, training=training)

    if sum(dim[2]) > 2:
        out = deconv2d(out, [2*dim[0], dim[1], dim[2]], scope="%s_conv_out"%scope, training=training, ema=ema, init=init)
    else:
        out = conv2d(out, [2*dim[0], dim[1], dim[2]], scope="%s_conv_out"%scope, training=training, ema=ema, init=init)

    h_stack1, h_stack2 = tf.split(out, 2, 3)
    sigmoid_out = tf.sigmoid(h_stack2)
    out = (h_stack1 * sigmoid_out)

    out_shp = out.get_shape().as_list()
    if out_shp[1:-1] < in_shp[1:-1]:
        x = tf.nn.avg_pool(x, [1, dim[2][0], dim[2][1], 1], strides=[1, dim[2][0], dim[2][1], 1], padding='SAME')
    elif out_shp[1:-1] > in_shp[1:-1]:
        warnings.warn(
            "The height and width of the output are larger than the input. There will be no residual connection.")
        residual = False


    if out_shp[-1] > in_shp[-1]:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, int(dim[0] - in_shp[-1])]])

    elif out_shp[-1] < in_shp[-1]:
        warnings.warn("The input has more feature maps than the output. There will be no residual connection.")
        residual = False

    if residual:
        out += x

    return out


def nin(x, num_units, **kwargs):
    s = tf.shape(x)
    sh = x.get_shape().as_list()
    x = tf.reshape(x, [tf.reduce_prod(s[:-1]), sh[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, [-1] + sh[1:-1] + [num_units])


def dense(x, num_units, scope="dense", training=True, ema=None, init=False, bias_initializer=tf.constant_initializer(0.)):
    with tf.variable_scope(scope):
        V = tf.get_variable('V', shape=[int(x.get_shape()[1]), num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,
                              initializer=bias_initializer, trainable=True)

        def maybe_avg(v):
            if ema is not None and not init:
                v = tf.cond(training, lambda: v, lambda: ema.average(v))
            return v

        if init:
            x = tf.matmul(x, tf.nn.l2_normalize(V.initialized_value(), 0))

            init_scale = .01
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.reshape(scale_init, [1, num_units]) * (x - tf.reshape(m_init, [1, num_units]))

        else:
            V = maybe_avg(V)
            g = maybe_avg(g)
            b = maybe_avg(b)

            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])


        return x


def sample_from_discretized_mix_logistic(l, nr_mix):
    """
    This function is copied from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py in reference to:
    See [Salimans et. al., 2017](https://arxiv.org/pdf/1701.05517)
    ([pdf](https://arxiv.org/pdf/1701.05517.pdf))

    log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval
    """
    ls = [-1] + l.get_shape().as_list()[1:]
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(
        tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)
    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)



