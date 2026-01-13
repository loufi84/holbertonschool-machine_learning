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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev_size = nx
        for i, layer_size in enumerate(layers, start=1):
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{i}"] = (
                np.random.randn(layer_size, prev_size) * np.sqrt(2 / prev_size)
            )
            self.__weights[f"b{i}"] = np.zeros((layer_size, 1))
            prev_size = layer_size

    @property
    def L(self):
        """
        Getter for the number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for the cache dictionary.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter for the weights dictionary.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        A function that calculates the forward propagation of the
        neural network.
        """
        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights[f"W{1}"]
            b = self.__weights[f"b{1}"]
            A_prev = self.__cache[f"A{i - 1}"]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache[f"A{i}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache
