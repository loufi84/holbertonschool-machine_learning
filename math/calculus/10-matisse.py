#!/usr/bin/env python3
"""
This module contains a function that calculates the derivative
of a polynomal.
"""


def poly_derivative(poly):
    """
    The said function.
    """
    # input validation
    if (not isinstance(poly, list) or
       len(poly) == 0 or
       not all(isinstance(n, (int, float)) for n in poly)):
        return None

    # check if the polynpmial is constant
    if len(poly) == 1:
        return [0]

    # compuyte the derivative
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

    # if the derivative is zeros
    if all(n == 0 for n in derivative):
        return [0]

    return derivative
