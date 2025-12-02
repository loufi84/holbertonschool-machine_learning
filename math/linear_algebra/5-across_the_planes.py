#!/usr/bin/env python3
"""
This module contains a function that add two matrices
element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    The function to add two matrices element-wise.
    """
    shape = __import__('2-size_me_please').matrix_shape
    if shape(mat1) != shape(mat2):
        return None
    if mat1.size == 0 or mat2.size == 0:
        return None
    new_matrix = []
    new_matrix = [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
    return new_matrix
