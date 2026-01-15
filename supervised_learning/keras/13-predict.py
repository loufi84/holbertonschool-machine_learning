#!/usr/bin/env python3
"""
This module contains a function that makes a prediction using
a neural network with Keras
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    The function that makes a prediction.
    """
    predictions = network.predict(
        x=data,
        verbose=verbose
    )

    return predictions
