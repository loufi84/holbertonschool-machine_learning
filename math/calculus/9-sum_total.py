#!/usr/bin/env python3
"""
This module contains a function that calculates the sum of i^2 from
i = 1 to n
"""


def summation_i_squared(n):
    """
    The function to calculate.
    """
    if type(n) is not int or n < 1:
        return None

    return n * (n + 1) * (2*n + 1) // 6
