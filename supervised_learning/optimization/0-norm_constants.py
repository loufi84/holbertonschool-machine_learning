#!/usr/bin/env python3
"""
This module contains a function that calculates the normalization
constant of a matrix.
"""
import numpy as np


def normalization_constants(X):
    """
    The function to calculate normalization.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
