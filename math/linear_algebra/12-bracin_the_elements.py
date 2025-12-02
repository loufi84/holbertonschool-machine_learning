#!/usr/bin/env python3
"""
A module that contains a numpy function that performs element-wise
operations.
"""


def np_elementwise(mat1, mat2):
    """
    The function that performs the operations.
    """
    sum = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return sum, sub, mul, div
