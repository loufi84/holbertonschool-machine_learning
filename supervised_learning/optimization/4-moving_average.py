#!/usr/bin/env python3
"""
This module contains a function that calculates the weighted moving average
of a dataset.
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a dataset using bias correction.
    """
    v = 0
    moving_averages = []

    for t, x in enumerate(data, start=1):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** t)
        moving_averages.append(v_corrected)

    return moving_averages
