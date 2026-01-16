"""
Custom loss functions for the classifier.
"""

import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package='AdClassifier')
def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred, label_smoothing=0.1):
    """
    Sparse categorical crossentropy with label smoothing.

    This is a compatibility function for loading older models that were saved
    with this custom loss function.

    Args:
        y_true: Ground truth labels (sparse format)
        y_pred: Predicted probabilities
        label_smoothing: Label smoothing factor

    Returns:
        Loss value
    """
    # Use Keras built-in sparse categorical crossentropy with label smoothing
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction='sum_over_batch_size'
    )
    return loss_fn(y_true, y_pred)
