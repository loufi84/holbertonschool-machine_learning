#!/usr/bin/env python3
"""
This module contains a function that shuffle datasets
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles two datasets in the same way.
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
