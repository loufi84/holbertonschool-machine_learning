#!/usr/bin/env python3
"""
This module contains 2 functions that saves and loads entire
models.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    The function tha save the model.
    """
    network.save(filename)


def load_model(filename):
    """
    The function that load the model.
    """
    return K.models.load_model(filename)
