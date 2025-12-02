#!/usr/bin/env python3
"""
This module contains a function that returns the transpose of a 2D matrix.
"""


def matrix_transpose(matrix):
    """
    The function that returns the transpose of a 2D matrix.
    """
    new_matrix = [[matrix[i][j] for i in range(len(matrix))]
                  for j in range(len(matrix[0]))]
    return new_matrix
