
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def log(x, base):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

def weight_variable(shape, name='weight', stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def rescale(array_x): # convert to [0,1]
    amax = tf.reduce_max(array_x, axis=1, keep_dims=True)
    amin = tf.reduce_min(array_x, axis=1, keep_dims=True)
    rescaled = array_x - amin
    rescaled = rescaled / amax
    return rescaled

# receives an array of images and return the mse per image.
# size ~ num of pixels in the img
def mse_array(array_x, array_y, size):
    rescale_x = array_x
    rescale_y = array_y
    se = tf.reduce_sum(tf.squared_difference(rescale_x, rescale_y), 1)
    inv_size = tf.to_float(1/size)
    return tf.scalar_mul(inv_size, se)


################ functions with tf_contrib_layers #################


def conv2d_bn_relu(x, w_size, num_outputs, keep_prob_, phase, scope): # output size should be the same.
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=tf.nn.relu,   # elu is an alternative
                                        normalizer_fn=tf.layers.batch_normalization,
                                        normalizer_params={'training': phase},
                                        scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d_bn_relu(x, w_size, num_outputs, stride, keep_prob_, phase, scope):
    conv_2d = tf.contrib.layers.conv2d_transpose(x, num_outputs, w_size,
                                                stride=stride,
                                                activation_fn=tf.nn.relu,   # elu is an alternative
                                                normalizer_fn=tf.layers.batch_normalization,
                                                normalizer_params={'training': phase},
                                                scope=scope)

    return tf.nn.dropout(conv_2d, keep_prob_)

def conv2d_relu(x, w_size, num_outputs, keep_prob_, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=tf.nn.relu,
                                        scope=scope)
    return tf.nn.dropout(conv_2d, keep_prob_)

def conv2d_bn(x, w_size, num_outputs, keep_prob_, phase, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=None,
                                        normalizer_fn=tf.layers.batch_normalization,
                                        normalizer_params={'training': phase},
                                        scope=scope)
    return tf.nn.dropout(conv_2d, keep_prob_)

def conv2d(x, w_size, num_outputs, keep_prob_, scope):
    conv_2d = tf.contrib.layers.conv2d(x, num_outputs, w_size,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope=scope)
    return tf.nn.dropout(conv_2d, keep_prob_)

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def concat(x1,x2):
    return tf.concat([x1, x2], 3)   
