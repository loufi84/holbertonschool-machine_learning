#!/usr/bin/env python3
"""
This module contains a function that builds a neural network
using the Keras library.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    The function to build the model using keras.
    """
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have the same length")

    model = K.Sequential()
    l2 = K.regularizers.l2(lambtha)
    dropout_rate = 1.0 - keep_prob

    for i, (units, act) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(K.layers.Dense(
                units=units,
                activation=act,
                kernel_regularizer=l2,
                input_shape=(nx,)
            ))
        else:
            model.add(K.layers.Dense(
                units=units,
                activation=act,
                kernel_regularizer=l2
            ))

        if i < len(layers) - 1:
            model.add(K.layers.Dropout(rate=dropout_rate))

    return model
