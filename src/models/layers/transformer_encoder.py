import tensorflow as tf
import keras
from ...utils.logging import LoggerMixin


@keras.saving.register_keras_serializable(package='AdClassifier', name='TransformerEncoder')
class TransformerEncoder(keras.layers.Layer, LoggerMixin):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        layer_norm_epsilon: float = 1e-6,
        use_causal_mask: bool = False,
        **kwargs
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_causal_mask = use_causal_mask

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.key_dim = embed_dim // num_heads

        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=dropout_rate
        )

        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation=activation),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(embed_dim)
        ])

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, attention_mask=None, training=None):
        if self.use_causal_mask:
            seq_len = tf.shape(inputs)[1]
            causal_mask = self._create_causal_mask(seq_len)
            attention_output = self.attention(
                inputs, inputs,
                attention_mask=causal_mask,
                training=training
            )
        else:
            attention_output = self.attention(
                inputs, inputs,
                attention_mask=attention_mask,
                training=training
            )

        attention_output = self.dropout1(attention_output, training=training)

        out1 = self.layernorm1(inputs + attention_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)

        return self.layernorm2(out1 + ffn_output)

    def _create_causal_mask(self, seq_len):
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'use_causal_mask': self.use_causal_mask
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
