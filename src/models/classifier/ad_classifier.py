import tensorflow as tf
import keras
from typing import Optional, Dict, Any, List
from pathlib import Path
from ..layers import TokenAndPositionEmbedding, MultiLayerTransformer, PoolingLayer
from .metrics import top_k_acc
from ...utils.logging import LoggerMixin
from ...utils.callbacks import create_wandb_callback


class AdClassifier(LoggerMixin):

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_length: int = 100,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 2048,
        dropout_rate: float = 0.1,
        embedding_dropout: float = None,
        attention_dropout: float = None,
        ffn_dropout: float = None,
        dense_dropout: float = None,
        pooling_strategy: str = 'cls',
        use_intermediate_dense: bool = True,
        intermediate_dim: int = None,
        activation: str = 'relu',
        use_class_weights: bool = False,
        label_smoothing: float = 0.0
    ):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.embedding_dropout = embedding_dropout if embedding_dropout is not None else dropout_rate
        self.attention_dropout = attention_dropout if attention_dropout is not None else dropout_rate
        self.ffn_dropout = ffn_dropout if ffn_dropout is not None else dropout_rate
        self.dense_dropout = dense_dropout if dense_dropout is not None else dropout_rate
        self.pooling_strategy = pooling_strategy
        self.use_intermediate_dense = use_intermediate_dense
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else embed_dim // 2
        self.activation = activation
        self.use_class_weights = use_class_weights
        self.label_smoothing = label_smoothing

        self.model = None
        self.class_weights = None
        self._is_compiled = False

    def build_model(self) -> keras.Model:
        inputs = keras.layers.Input(shape=(self.max_length,), name='input_ids')

        embeddings = TokenAndPositionEmbedding(
            maxlen=self.max_length,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            dropout_rate=self.embedding_dropout,
            name='embeddings'
        )(inputs)

        transformer_output = MultiLayerTransformer(
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            activation=self.activation,
            name='transformer'
        )(embeddings)

        pooled_output = PoolingLayer(
            pooling_strategy=self.pooling_strategy,
            embed_dim=self.embed_dim if self.pooling_strategy == 'attention' else None,
            name='pooling'
        )(transformer_output)

        if self.use_intermediate_dense:
            dense_output = keras.layers.Dense(
                self.intermediate_dim,
                activation='relu',
                name='dense_representation'
            )(pooled_output)

            dense_output = keras.layers.Dropout(self.dense_dropout)(dense_output)

            outputs = keras.layers.Dense(
                self.num_classes,
                activation='softmax',
                name='classification'
            )(dense_output)
        else:
            outputs = keras.layers.Dense(
                self.num_classes,
                activation='softmax',
                name='classification'
            )(pooled_output)

        model = keras.Model(inputs=inputs, outputs=outputs, name='ad_classifier')

        self.model = model
        return model

    def compile_model(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        optimizer: str = 'adamw',
        loss: str = 'sparse_categorical_crossentropy',
        metrics: Optional[List[str]] = None
    ) -> None:
        if self.model is None:
            raise RuntimeError("Model must be built before compilation")

        if optimizer.lower() == 'adamw':
            opt = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                clipnorm=1.0
            )
        elif optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        if self.label_smoothing > 0 and loss == 'sparse_categorical_crossentropy':
            import tensorflow as tf

            def sparse_categorical_crossentropy_with_smoothing(y_true, y_pred):
                num_classes = tf.shape(y_pred)[-1]
                y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

                smoothing = self.label_smoothing
                y_true_smooth = y_true_one_hot * (1 - smoothing) + smoothing / tf.cast(num_classes, tf.float32)

                return keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

            loss_fn = sparse_categorical_crossentropy_with_smoothing
        else:
            loss_fn = loss

        if metrics is None:
            metrics = ['sparse_categorical_accuracy']

        metrics.append(top_k_acc)

        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics
        )

        self._is_compiled = True
        self.logger.info(f"Model compiled with {optimizer} optimizer and {loss} loss")

    def set_class_weights(self, class_weights: Dict[int, float]) -> None:
        self.class_weights = class_weights
        self.logger.info(f"Set class weights for {len(class_weights)} classes")

    def get_model_summary(self) -> str:
        if self.model is None:
            return "Model not built yet"

        import io
        summary_io = io.StringIO()
        self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        return summary_io.getvalue()

    def save_model(self, save_path: str, save_format: str = 'keras') -> None:
        if self.model is None:
            raise RuntimeError("Model must be built before saving")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format == 'keras':
            if not str(save_path).endswith('.keras'):
                save_path = Path(str(save_path) + '.keras')
            self.model.save(save_path)
        else:
            self.model.export(save_path)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: str) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = keras.models.load_model(model_path)
        self._is_compiled = True
        self.logger.info(f"Model loaded from {model_path}")

    def predict(self, inputs, batch_size: int = 32) -> tf.Tensor:
        if self.model is None:
            raise RuntimeError("Model must be built before prediction")

        return self.model.predict(inputs, batch_size=batch_size)

    def predict_classes(self, inputs, batch_size: int = 32) -> tf.Tensor:
        predictions = self.predict(inputs, batch_size)
        return tf.argmax(predictions, axis=-1)

    def predict_top_k(self, inputs, k: int = 3, batch_size: int = 32) -> tuple:
        predictions = self.predict(inputs, batch_size)
        top_k_probs, top_k_indices = tf.nn.top_k(predictions, k=k)
        return top_k_indices, top_k_probs

    def get_config(self) -> Dict[str, Any]:
        return {
            'vocab_size': self.vocab_size,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'embedding_dropout': self.embedding_dropout,
            'attention_dropout': self.attention_dropout,
            'ffn_dropout': self.ffn_dropout,
            'dense_dropout': self.dense_dropout,
            'pooling_strategy': self.pooling_strategy,
            'use_intermediate_dense': self.use_intermediate_dense,
            'intermediate_dim': self.intermediate_dim,
            'activation': self.activation,
            'use_class_weights': self.use_class_weights,
            'label_smoothing': self.label_smoothing
        }

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs: int = 10,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        wandb_config: Optional[Any] = None,
        verbose: int = 1,
        **fit_kwargs
    ):
        if self.model is None:
            raise RuntimeError("Model must be built before training")

        if not self._is_compiled:
            raise RuntimeError("Model must be compiled before training")

        callbacks = callbacks or []

        if wandb_config is not None:
            wandb_callback = create_wandb_callback(
                config=self.get_config(),
                wandb_config=wandb_config
            )
            if wandb_callback is not None:
                callbacks.append(wandb_callback)
                self.logger.info("W&B callback added to training")

        fit_args = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_data': (x_val, y_val),
            'callbacks': callbacks,
            'verbose': verbose,
            'shuffle': True
        }

        if self.class_weights is not None:
            fit_args['class_weight'] = self.class_weights

        fit_args.update(fit_kwargs)

        self.logger.info(f"Starting training for {epochs} epochs")
        history = self.model.fit(x_train, y_train, **fit_args)

        self.logger.info("Training completed")
        return history
