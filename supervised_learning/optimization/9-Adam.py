#!/usr/bin/env python3
"""
This module contains a function that updates a variable using the
Adam optimization.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.
    """
    # Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Compute bias-corrected moments
    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)

    # Update parameters
    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)

    return var, v, s
