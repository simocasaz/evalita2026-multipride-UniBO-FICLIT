"""multipride package public API.

Re-export commonly used helpers so notebooks and external code can
import them as `from multipride import clean_text` instead of
reaching into submodules.
"""

from .preprocessing import (
    clean_text,
    load_tokenizer,
    tokenize_batch,
    tokenize_batch_with_bios,
)
from .data_utils import load_split, augment_data
from .training_utils import run_hyperparameter_search, train_save_best_model
from .inference import run_inference
from .evaluation import compute_metrics

__all__ = [
    "clean_text",
    "load_tokenizer",
    "tokenize_batch",
    "tokenize_batch_with_bios",
    "load_split",
    "augment_data",
    "run_hyperparameter_search",
    "train_save_best_model",
    "run_inference",
    "compute_metrics",
]
