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
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def get_W(self):
        return self.__W

    def get_b(self):
        return self.__b

    def get_A(self):
        return self.__A
