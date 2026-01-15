#!/usr/bin/env python3
"""
This module contains 2 functions that save and load model's
weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Save the model weight.
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load the model weight.
    """
    network.load_weights(filename)
