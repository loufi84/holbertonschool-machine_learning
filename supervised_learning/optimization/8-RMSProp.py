#!/usr/bin/env python3
"""
This module contains a function that sets up the RMSProp optimizer
using tensorflow.
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates a TensorFlow RMSProp optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
