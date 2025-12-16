import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package='AdClassifier', name='top_k_acc')
def top_k_acc(y_true, y_pred, k=3):
    top_k_preds = tf.math.top_k(y_pred, k=k).indices
    y_true = tf.cast(y_true, tf.int32)
    top_k_preds = tf.cast(top_k_preds, tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true[:, None], top_k_preds), tf.float32), axis=1)
    return tf.reduce_mean(correct)
