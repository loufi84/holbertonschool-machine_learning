#!/usr/bin/env python3
"""
This module contains the function one_hot using Keras library.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    The function to one_hot.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
