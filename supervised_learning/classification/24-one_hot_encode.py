#!/usr/bin/env python3
"""
That module contains a function that converts a numeric label
vector into a one-hot matrix.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    The method that converts the numeric label.
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None

    if not isinstance(classes, int) or classes <= 0:
        return None

    m = Y.shape[0]

    if np.any(Y < 0) or np.any(Y >= classes):
        return None

    one_hot = np.zeros((classes, m))

    for i in range(m):
        one_hot[Y[i], i] = 1

    return one_hot
