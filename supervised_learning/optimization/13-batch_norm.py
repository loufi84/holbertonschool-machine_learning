#!/usr/bin/env python3
"""
This module contains a function that normalizes an unactivated output
of a neural network, using batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization.
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(var + epsilon)

    return gamma * Z_norm + beta
