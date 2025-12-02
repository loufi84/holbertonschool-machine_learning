#!/usr/bin/env python3
import numpy as np
"""
A module that contains a function that concatenates two matrices along
a specific axis.

"""


def np_cat(mat1, mat2, axis=0):
    """
    The function that concatenates two matrices along a
    specified axis.

    """
    new_matrix = np.concatenate((mat1, mat2), axis)
    return new_matrix
