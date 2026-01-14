#!/usr/bin/env python3
"""
This module contains a function that uses Adam optimization
in the Keras library.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    The function to optimize the model.
    """
    optimizer ! K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy)=']
    )
