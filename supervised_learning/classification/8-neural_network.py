#!/usr/bin/env python3
"""
This modulecontains the class definition of a base neural network.
"""
import numpy as np


class NeuralNetwork:
    """
    The class to define the base neural network.

    nx: number of input features.
    nodes: number of nodes in the hidden layer.
    W1: weights of hidden layer.
    W2: weights vector for output neuron.
    b1: bias for hidden layer.
    b2: bias for output neuron.
    A1: activated output for hidden layer.
    A2: activated output for output neuron.
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.W2 = np.random.randn(1, nodes)
        self.b1 = np.zeros((nodes, 1))
        self.b2 = 0
        self.A1 = 0
        self.A2 = 0
