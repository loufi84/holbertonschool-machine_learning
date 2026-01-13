#!/usr/bin/env python3
"""
This module contains the class definition of a deep
neural network that performs binary classification.
"""
import numpy as np
import matplotlib as plt
import pickle


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
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]
            A_prev = self.__cache[f"A{i - 1}"]

            Z = np.dot(W, A_prev) + b

            if i == self.__L:
                Z_shift = Z - np.max(Z, axis=0, keepdims=True)
                expZ = np.exp(Z_shift)
                A = expZ / np.sum(expZ, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))

            self.__cache[f"A{i}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        A function that calculates the cost of the model using logic
        regression.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        The function that evaluates the neural network predictions.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        classes, m = Y.shape
        preds = np.zeros((classes, m))

        pred_labels = np.argmax(A, axis=0)
        preds[pred_labels, np.arange(m)] = 1

        return preds, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        This method calculates one pass of gradient descent of
        the neural network.
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()

        A_L = cache[f"A{self.__L}"]
        dZ = A_L - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache[f"A{i - 1}"]
            W = weights_copy[f"W{i}"]

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{i}"] = self.__weights[f"W{i}"] - alpha * dW
            self.__weights[f"b{i}"] = self.__weights[f"b{i}"] - alpha * db

            if i > 1:
                A_prev_act = cache[f"A{i - 1}"]
                dA_prev = np.dot(W.T, dZ)
                dZ = dA_prev * (A_prev_act * (1 - A_prev_act))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if (verbose or graph):
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    iters.append(i)
                    costs.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iters, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled object and return the loaded object.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, OSError):
            return None
