from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
import tensorflow.contrib as contrib
from Regularizers.layers_tf import *


def dncnn(input, output_channels=1):
    input_shape_of_conv_layer = []
    with tf.variable_scope('block1'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 6 + 1):
        with tf.variable_scope('block%d' % layers):
            input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=True)
            output = tf.nn.relu(output)
    with tf.variable_scope('block7'):
        input_shape_of_conv_layer.append([input.shape[1],input.shape[2]])
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return output, input_shape_of_conv_layer
