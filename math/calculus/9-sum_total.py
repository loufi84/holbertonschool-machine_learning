#!/usr/bin/env python3
"""
This module contains a function that calculates the sum of i^2 from
i = 1 to n
"""


def summation_i_squared(n):
    """
    
    """
    if not isinstance(n, int):
        return None
    if n == 1:
        return 1
    return n*n + summation_i_squared(n - 1)
