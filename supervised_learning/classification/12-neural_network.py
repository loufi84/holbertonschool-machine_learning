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

        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter function for W1.
        """
        return self.__W1

    @property
    def W2(self):
        """
        Getter function for W2.
        """
        return self.__W2

    @property
    def b1(self):
        """
        Getter function for b1.
        """
        return self.__b1

    @property
    def b2(self):
        """
        Getter function for b2.
        """
        return self.__b2

    @property
    def A1(self):
        """
        Getter function for A1.
        """
        return self.__A1

    @property
    def A2(self):
        """
        Getter function for A2.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        A function that calculate the forward propagation of the
        neural network.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        A function that calculate the cost of the model, using
        logistice regression.
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """
        A function that evaluates the neural network's
        prediction.

        Return:
        The cost and the predictions.
        """
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = np.where(A >= 0.5).astype(int)
        return prediction, cost
