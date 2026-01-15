#!/usr/bin/env python3
"""
This module contains a function that trains a neural network
with Keras.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    The function to train the model.
    """

    callbacks = []

    if early_stopping and validation_data is not None:
        callbacks.append(
            K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                mode='min'
            )
        )

    if learning_rate_decay and validation_data is not None:
        def schedule(epoch):
            """
            Inverse time decay.
            """
            lr = alpha / (1 + decay_rate * epoch)
            print(f"Learning rate updated to {lr}")
            return lr

        callbacks.append(K.callbacks.LearningRateScheduler(schedule,
                         verbose=0))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
