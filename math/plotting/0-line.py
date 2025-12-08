#!/usr/bin/env python3
"""
This module provides a function that draw a line graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    The function to draw the graph.
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)

    plt.show()
