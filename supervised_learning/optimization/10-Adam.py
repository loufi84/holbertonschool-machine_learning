#!/usr/bin/env python3
"""
This module contains a function that creates an Adam optimizer
using tensorflow.
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates a TensorFlow Adam optimizer.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
