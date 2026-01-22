#!/usr/bin/env python3
"""
This module contains a function that creates a batch normalization layer
for a neural network, using tensorflow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer in TensorFlow.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        use_bias=False
    )(prev)

    gamma = tf.Variable(
        initial_value=tf.ones((1, n)),
        trainable=True
    )
    beta = tf.Variable(
        initial_value=tf.zeros((1, n)),
        trainable=True
    )

    mean, variance = tf.nn.moments(dense, axes=[0])
    Z_norm = (dense - mean) / tf.sqrt(variance + 1e-7)
    Z_tilde = gamma * Z_norm + beta

    return activation(Z_tilde)
