# Ref: https://github.com/arashsaber/Deep-Convolutional-AutoEncoder/blob/master/ConvolutionalAutoEncoder.py
import tensorflow.compat.v1 as tf
import numpy as np

#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 1, 1, 1], pad='SAME'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.initializers.glorot_normal)
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.initializers.glorot_normal)
        out = tf.nn.conv2d(input,W,strides=strides, padding=pad)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1], pad='SAME'):
    with tf.variable_scope(name):
        out = tf.layers.conv2d_transpose(input,
                                        filters = n_outputs,
                                        kernel_size=kshape,
                                        strides=strides,
                                        padding=pad,
                                        kernel_initializer=tf.initializers.glorot_normal,
                                        bias_initializer=tf.initializers.glorot_normal,
                                        activation=tf.nn.relu)
        return out
#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.variable_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out
#   ---------------------------------
def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.variable_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out
#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.variable_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.initializers.glorot_normal)
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.initializers.glorot_normal)
        input = tf.reshape(input, [-1, input_size])
        out = tf.add(tf.matmul(input, W), b)
        return out
#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.variable_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out
#   ---------------------------------