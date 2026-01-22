#!/usr/bin/env python3
"""
This module contains a function that creates a learning rate decay
operation in tensorflow.
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a TensorFlow inverse time decay learning rate schedule.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
