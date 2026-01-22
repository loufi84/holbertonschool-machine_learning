#!/usr/bin/env python3
"""
This module contains a function that normalize a matrix
"""
import numpy as np


def normalize(X, m, s):
    """
    The function to standardize a matrix
    """
    return (X - m) / s
