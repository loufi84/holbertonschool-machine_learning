#!/usr/bin/env python3
"""
This module contains a function that calculates the precision for each
class in a confusion matrix.
"""
import numpy as np


def precision(confusion):
    """
    The function that calculates the precision.
    """
    true_pos = np.diag(confusion)
    pred_pos = np.sum(confusion, axis=0)

    precision = true_pos / pred_pos

    return precision
