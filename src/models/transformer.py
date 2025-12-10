import tensorflow as tf
import keras
from typing import Optional
from ..utils import LoggerMixin


@keras.saving.register_keras_serializable(package='AdClassifier', name='TokenAndPositionEmbedding')
class TokenAndPositionEmbedding(keras.layers.Layer, LoggerMixin):
    """Enhanced token and position embedding layer."""

    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        mask_zero: bool = True,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize token and position embedding layer.

        Args:
            maxlen: Maximum sequence length
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            mask_zero: Whether to mask zero values (padding)
            dropout_rate: Dropout rate for embeddings
            **kwargs: Additional layer arguments
        """
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
        """Forward pass."""
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)

        token_embeddings = self.token_emb(x)
        position_embeddings = self.pos_emb(positions)

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings, training=training)

    def get_config(self):
        """Get layer configuration."""
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
        """Create layer from configuration."""
        return cls(**config)


@keras.saving.register_keras_serializable(package='AdClassifier', name='TransformerEncoder')
class TransformerEncoder(keras.layers.Layer, LoggerMixin):
    """Enhanced transformer encoder layer with improved functionality."""

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
        """
        Initialize transformer encoder layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
            activation: Activation function for feed-forward network
            layer_norm_epsilon: Epsilon for layer normalization
            use_causal_mask: Whether to use causal (autoregressive) masking
            **kwargs: Additional layer arguments
        """
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
        """Forward pass."""
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
        """Create causal mask for autoregressive attention."""
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return mask[tf.newaxis, tf.newaxis, :, :]

    def get_config(self):
        """Get layer configuration."""
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
        """Create layer from configuration."""
        return cls(**config)

@keras.saving.register_keras_serializable(package='AdClassifier', name='MultiLayerTransformer')
class MultiLayerTransformer(keras.layers.Layer, LoggerMixin):
    """Multi-layer transformer encoder stack."""

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        **kwargs
    ):
        """
        Initialize multi-layer transformer.

        Args:
            num_layers: Number of transformer layers
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout_rate: Dropout rate
            activation: Activation function
            **kwargs: Additional arguments
        """
        super(MultiLayerTransformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.transformer_layers = [
            TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                name=f'transformer_layer_{i}'
            )
            for i in range(num_layers)
        ]

    def call(self, inputs, attention_mask=None, training=None):
        """Forward pass through all transformer layers."""
        x = inputs

        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask, training=training)

        return x

    def get_config(self):
        """Get layer configuration."""
        config = super(MultiLayerTransformer, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)

@keras.saving.register_keras_serializable(package='AdClassifier', name='PoolingLayer')
class PoolingLayer(keras.layers.Layer):
    """Enhanced pooling layer for sequence classification."""

    def __init__(
        self,
        pooling_strategy: str = 'cls',
        embed_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize pooling layer.

        Args:
            pooling_strategy: Strategy for pooling ('cls', 'mean', 'max', 'attention')
            embed_dim: Embedding dimension (required for attention pooling)
            **kwargs: Additional layer arguments
        """
        super(PoolingLayer, self).__init__(**kwargs)
        self.pooling_strategy = pooling_strategy
        self.embed_dim = embed_dim

        if pooling_strategy == 'attention' and embed_dim is None:
            raise ValueError("embed_dim is required for attention pooling")

        if pooling_strategy == 'attention':
            self.attention_weights = keras.layers.Dense(1, activation='tanh')
            self.attention_softmax = keras.layers.Softmax(axis=1)

    def call(self, inputs, mask=None):
        """Forward pass."""
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
        """Get layer configuration."""
        config = super(PoolingLayer, self).get_config()
        config.update({
            'pooling_strategy': self.pooling_strategy,
            'embed_dim': self.embed_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)