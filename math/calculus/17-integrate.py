#!/usr/bin/env python3
"""
Ths module contains a function that calculate the integral of
a polynomial.
"""


def poly_integral(poly, C=0):
    """
    The said function.
    """
    # input validation
    if not isinstance(poly, list) or not isinstance(C, int):
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None
        
    # integration constant
    integral = [C]

    # coefficient integration
    for i, coeff in enumerate(poly):
        new_coeff = coeff / (i + 1)

        # integer conversion
        if new_coeff.is_integer():
            new_coeff = int(new_coeff)

        integral.append(new_coeff)

    # remove trailing zeros
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
