#!/usr/bin/env python3
"""
A module that contains a single function that uses NumPy to transposes
a matrix.
"""


def np_transpose(matrix):
    """
    The function to transposes the matrix using NumPy.
    """
    new_matrix = matrix.T
    return new_matrix
