import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package='AdClassifier')
def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred, label_smoothing=0.1):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction='sum_over_batch_size'
    )
    return loss_fn(y_true, y_pred)
