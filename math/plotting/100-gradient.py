#!/usr/bin/env python3
"""
This module contains a function that create a scatter plot of sampled
elevations on a mountain.
"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    The function that create the scatter plot.
    """
    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    # your code here
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    plt.title("Mountain Elevation")

    sc = plt.scatter(x, y, c=z, cmap="viridis")

    char = plt.colorbar(sc)
    char.set_label("elevation (m)")

    plt.show()
