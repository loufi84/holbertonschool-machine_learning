#!/usr/bin/env python3
"""
This module contains a function that tests a neural
network.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    The function that tests the model.
    """
    results = network.evaluates(
        x=data,
        y=labels,
        verbose=verbose
    )

    return results
