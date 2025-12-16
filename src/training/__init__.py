"""Training scripts for the classifier model."""

from .train_model import main as train
from .train_model_wandb import main as train_wandb
from .train_model_wandb_gpu import main as train_wandb_gpu

__all__ = ['train', 'train_wandb', 'train_wandb_gpu']
