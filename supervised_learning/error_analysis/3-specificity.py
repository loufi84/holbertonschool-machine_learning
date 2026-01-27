#!/usr/bin/env python3
"""
This module contains that contains a function that calculates the specificity
for each class in a confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """
    The function taht calculates the specificity.
    """
    total = np.sum(confusion)

    true_pos = np.diag(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos

    true_neg = total - (true_pos + false_pos + false_neg)

    specificity = true_neg / (true_neg + false_pos)

    return specificity
