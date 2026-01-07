#!/usr/bin/env python3
"""
This module contains the definition of the class Neuron
"""
import numpy as np


class Neuron:
    """
    The definition of the class Neuron
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter for __W
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for __b
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for __A
        """
        return self.__A
