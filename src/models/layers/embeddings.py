import tensorflow as tf
import keras
from ...utils.logging import LoggerMixin


@keras.saving.register_keras_serializable(package='AdClassifier', name='TokenAndPositionEmbedding')
class TokenAndPositionEmbedding(keras.layers.Layer, LoggerMixin):

    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        mask_zero: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.mask_zero = mask_zero
        self.dropout_rate = dropout_rate

        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=mask_zero
        )

        self.pos_emb = keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

        self.dropout = keras.layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)

        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings, training=training)

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'mask_zero': self.mask_zero,
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
