#!/usr/bin/env python3
"""
This module contains the class definition of a deep
neural network that performs binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    The class that defines the neural network.
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]

            if l == 1:
                prev_size = nx
            else:
                prev_size = layers[l - 2]

            self.weights[f"W{1}"] = (
                np.random.randn(layer_size, prev_size)
                * np.sqrt(2 / prev_size)
            )
            self.weights[f"b{1}"] = np.zeros((layer_size, 1))
