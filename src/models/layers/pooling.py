import tensorflow as tf
import keras
from typing import Optional


@keras.saving.register_keras_serializable(package='AdClassifier', name='PoolingLayer')
class PoolingLayer(keras.layers.Layer):

    def __init__(
        self,
        pooling_strategy: str = 'cls',
        embed_dim: Optional[int] = None,
        **kwargs
    ):
        super(PoolingLayer, self).__init__(**kwargs)
        self.pooling_strategy = pooling_strategy
        self.embed_dim = embed_dim

        if pooling_strategy == 'attention' and embed_dim is None:
            raise ValueError("embed_dim is required for attention pooling")

        if pooling_strategy == 'attention':
            self.attention_weights = keras.layers.Dense(1, activation='tanh')
            self.attention_softmax = keras.layers.Softmax(axis=1)

    def call(self, inputs, mask=None):
        if self.pooling_strategy == 'cls':
            return inputs[:, 0, :]

        elif self.pooling_strategy == 'mean':
            if mask is not None:
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                masked_inputs = inputs * mask
                sum_embeddings = tf.reduce_sum(masked_inputs, axis=1)
                sum_mask = tf.reduce_sum(mask, axis=1)
                return sum_embeddings / (sum_mask + 1e-9)
            else:
                return tf.reduce_mean(inputs, axis=1)

        elif self.pooling_strategy == 'max':
            return tf.reduce_max(inputs, axis=1)

        elif self.pooling_strategy == 'attention':
            attention_scores = self.attention_weights(inputs)
            attention_weights = self.attention_softmax(attention_scores)

            if mask is not None:
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                attention_weights = attention_weights * mask
                attention_weights = attention_weights / (tf.reduce_sum(attention_weights, axis=1, keepdims=True) + 1e-9)

            return tf.reduce_sum(inputs * attention_weights, axis=1)

        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")

    def get_config(self):
        config = super(PoolingLayer, self).get_config()
        config.update({
            'pooling_strategy': self.pooling_strategy,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
