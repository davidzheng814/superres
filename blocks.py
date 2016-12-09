import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib.layers.python.layers.layers import batch_norm

import config as cfg

def relu_block(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def res_block(inp, is_training_cond, reuse=False):
    """ResNet B Block.
    TODO: decide whether bias is worth it
    """
    inp_shape = inp.get_shape()
    assert inp_shape[-1] == 64 # Must have 64 channels to start.

    with tf.variable_scope("resconv1"):
        h = conv_block(inp, relu=True, bn=True, is_training_cond=is_training_cond,
                reuse=reuse)

    with tf.variable_scope("resconv2"):
        h = conv_block(h, bn=True, is_training_cond=is_training_cond, reuse=reuse)

    return inp + h

def deconv_block(inp, relu=True, output_channels=64):
    """
    TODO: Decide whether or not to add bias.
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    """
    inp_shape = inp.get_shape()
    kernel_shape = (3, 3, output_channels, inp_shape[-1])
    output_shape = (int(inp_shape[0]), int(inp_shape[1] * 2), int(inp_shape[2] * 2), output_channels)
    strides = [1, 2, 2, 1]

    weights = tf.get_variable('weights', kernel_shape,
        initializer=tf.random_normal_initializer(stddev=0.02))
    h = tf.nn.conv2d_transpose(inp, weights, output_shape, strides)

    if relu:
        h = tf.nn.relu(h)

    return h

def conv_block(inp, relu=False, leaky_relu=False, bn=False,
               output_channels=64, stride=1, is_training_cond=None,
               reuse=False):
    inp_shape = inp.get_shape()
    kernel_shape = (3, 3, inp_shape[-1], output_channels)
    strides = [1, stride, stride, 1]

    weights = tf.get_variable('weights', kernel_shape,
        initializer=tf.random_normal_initializer(stddev=0.02))
    h = tf.nn.conv2d(inp, weights, strides, padding='SAME')

    if leaky_relu:
        h = relu_block(h, alpha=0.01)

    if bn:
        h = batch_norm(h, reuse=reuse, is_training=is_training_cond,
            scope=tf.get_variable_scope(), scale=True)

    if relu:
        h = tf.nn.relu(h)

    return h

def dense_block(inp, leaky_relu=False, sigmoid=False,
                output_size=1024):
    inp_size = inp.get_shape()
    h = tf.reshape(inp, [int(inp_size[0]), -1])
    h_size = h.get_shape()[1]

    w = tf.get_variable("w", [h_size, output_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [output_size],
        initializer=tf.constant_initializer(0.0))

    h = tf.matmul(h, w) + b

    if leaky_relu:
        h = relu_block(h, alpha=0.01)

    if sigmoid:
        h = tf.nn.sigmoid(h)

    return h

