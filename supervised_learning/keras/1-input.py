#!/usr/bin/env python3
"""
This module contains a function taht create a model using Keras
library, without the use of Sequential function
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    The function that creates the model.
    """
    if len(layers) != len(activations):
        raise ValueError("layers and activations must have the same length")

    l2 = K.regularizers.l2(lambtha)
    drop_rate = 1.0 - keep_prob

    inputs = K.Input(shape=(nx,))
    x = inputs

    for i, (units, act) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=units,
            activation=act,
            kernel_regularizer=l2
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=drop_rate)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
