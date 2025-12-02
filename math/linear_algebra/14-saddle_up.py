#!/usr/bin/env python3
"""
A module that contains a function that performs matrix multiplication.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """
    The function that performs the multiplication.
    """
    new_array = np.dot(mat1, mat2)
    return new_array
