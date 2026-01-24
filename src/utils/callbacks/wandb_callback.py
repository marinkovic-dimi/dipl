import keras
from pathlib import Path
from typing import Optional, Dict, Any

from ..logging import get_logger

logger = get_logger(__name__)


class WandbCallback(keras.callbacks.Callback):

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        project: str = "ad-classifier",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        log_model: bool = True,
        log_gradients: bool = False,
        log_frequency: int = 100,
        save_code: bool = True
    ):
        super().__init__()
        self.config = config or {}
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.notes = notes
        self.log_model = log_model
        self.log_gradients = log_gradients
        self.log_frequency = log_frequency
        self.save_code = save_code

        self.wandb = None
        self.batch_count = 0

        try:
            import wandb
            self.wandb = wandb
            logger.info("Weights & Biases imported successfully")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb = None

    def on_train_begin(self, logs=None):
        if self.wandb is None:
            logger.warning("W&B callback disabled - wandb not installed")
            return

        try:
            self.wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                notes=self.notes,
                config=self.config,
                save_code=self.save_code,
                reinit=True
            )

            if self.model and hasattr(self.model, 'summary'):
                logger.info("Logging model summary to W&B")
                import io
                summary_io = io.StringIO()
                self.model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
                model_summary = summary_io.getvalue()

                self.wandb.config.update({
                    'model_summary': model_summary,
                    'total_params': self.model.count_params()
                })

            logger.info(f"W&B run initialized: {self.wandb.run.name}")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.wandb = None

    def on_epoch_end(self, epoch, logs=None):
        if self.wandb is None or not self.wandb.run:
            return

        logs = logs or {}

        metrics = {'epoch': epoch + 1}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics[f'epoch/{key}'] = value

        try:
            self.wandb.log(metrics)
            logger.debug(f"Logged epoch {epoch + 1} metrics to W&B")
        except Exception as e:
            logger.error(f"Failed to log epoch metrics: {e}")

    def on_batch_end(self, batch, logs=None):
        if self.wandb is None or not self.wandb.run:
            return

        self.batch_count += 1

        if self.batch_count % self.log_frequency == 0:
            logs = logs or {}

            batch_metrics = {'batch': self.batch_count}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    batch_metrics[f'batch/{key}'] = value

            try:
                self.wandb.log(batch_metrics)
            except Exception as e:
                logger.error(f"Failed to log batch metrics: {e}")

    def on_train_end(self, logs=None):
        if self.wandb is None or not self.wandb.run:
            return

        try:
            if self.log_model and self.model:
                model_path = Path("wandb_model_checkpoint.keras")
                self.model.save(model_path)

                self.wandb.save(str(model_path))
                logger.info(f"Saved model to W&B: {model_path}")

                if model_path.exists():
                    model_path.unlink()

            self.wandb.finish()
            logger.info("W&B run finished")

        except Exception as e:
            logger.error(f"Error finishing W&B run: {e}")


def create_wandb_callback(
    config: Optional[Dict[str, Any]] = None,
    wandb_config: Optional[Any] = None
) -> Optional[WandbCallback]:
    if wandb_config is None:
        return None

    if hasattr(wandb_config, '__dict__'):
        wandb_dict = wandb_config.__dict__
    else:
        wandb_dict = wandb_config

    if not wandb_dict.get('enabled', False):
        logger.info("W&B logging disabled in configuration")
        return None

    return WandbCallback(
        config=config,
        project=wandb_dict.get('project', 'ad-classifier'),
        entity=wandb_dict.get('entity'),
        name=wandb_dict.get('name'),
        tags=wandb_dict.get('tags', []),
        notes=wandb_dict.get('notes'),
        log_model=wandb_dict.get('log_model', True),
        log_gradients=wandb_dict.get('log_gradients', False),
        log_frequency=wandb_dict.get('log_frequency', 100),
        save_code=wandb_dict.get('save_code', True)
    )
