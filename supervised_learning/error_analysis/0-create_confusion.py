#!/usr/bin/env python3
"""
This module contains a function that create a confusion matrix.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    The function that creates the confusion matrix.
    """
    # Ici, le nombre de classes
    classes = labels.shape[1]

    # Conversion du one-hot en indices
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)

    # Création de la matrice de confusion
    conf = np.zeros((classes, classes), dtype=int)

    # Incrémentation des bonnes cases
    np.add.at(conf, (y_true, y_pred), 1)

    return conf
