import keras
from .transformer_encoder import TransformerEncoder
from ...utils.logging import LoggerMixin


@keras.saving.register_keras_serializable(package='AdClassifier', name='MultiLayerTransformer')
class MultiLayerTransformer(keras.layers.Layer, LoggerMixin):

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        attention_dropout: float = None,
        ffn_dropout: float = None,
        activation: str = 'relu',
        **kwargs
    ):
        super(MultiLayerTransformer, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout_rate
        self.ffn_dropout = ffn_dropout if ffn_dropout is not None else dropout_rate
        self.activation = activation

        self.transformer_layers = [
            TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                activation=activation,
                name=f'transformer_layer_{i}'
            )
            for i in range(num_layers)
        ]

    def call(self, inputs, attention_mask=None, training=None):
        x = inputs

        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask, training=training)

        return x

    def get_config(self):
        config = super(MultiLayerTransformer, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'attention_dropout': self.attention_dropout,
            'ffn_dropout': self.ffn_dropout,
            'activation': self.activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
