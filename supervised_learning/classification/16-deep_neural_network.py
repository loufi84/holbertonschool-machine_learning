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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev_size = nx
        for l, layer_size in enumerate(layers, start=1):
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.weights[f"W{l}"] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.weights[f"b{l}"] = np.zeros((layer_size, 1))
            prev_size = layer_size
