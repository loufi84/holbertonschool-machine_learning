#!/usr/bin/env python3
"""
This module contains a function that creates a momentum optimizer
using momentum.
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow Momentum optimizer.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
