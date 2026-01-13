#!/usr/bin/env python3
"""
This module contains a function that converts a one_hot matrix
into a vectore of label.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    The function that decodes the one_hot matrix.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    classes, m = one_hot.shape

    if classes == 0 or m == 0:
        return None

    return np.argmax(one_hot, axis=0)
