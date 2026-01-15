#!/usr/bin/env python3
"""
This module contains functions to save and load a Keras model configuration.
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format.
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model from a JSON configuration file.
    """
    with open(filename, 'r') as f:
        config = f.read()

    return K.models.model_from_json(config)
