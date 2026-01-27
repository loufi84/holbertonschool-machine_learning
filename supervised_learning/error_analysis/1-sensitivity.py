#!/usr/bin/env python3
"""
This module contains a function that calculates the sensitivity (accuracy)
for each class in a confusion matrix.
"""
import numpy as np


def sensitivity(confusion):
    """
    The function that calculates the sensitivity.
    """
    true_pos = np.diag(confusion)
    actual_pos = np.sum(confusion, axis=1)

    sensitivity = true_pos / actual_pos

    return sensitivity
