#!/usr/bin/env python3
"""
This modules contain a method that calculates the F1 score
of the confusion matrix.
"""
import numpy as np


def f1_score(confusion):
    """
    The method that calculates the F1 score.
    """
    sensitivity = __import__('1-sensitivity').sensitivity(confusion)
    precision = __import__('2-precision').precision(confusion)

    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    return f1
