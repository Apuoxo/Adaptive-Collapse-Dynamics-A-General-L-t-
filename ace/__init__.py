"""
Adaptive Collapse Dynamics: A General L²/t Principle from Gene Regulation to Transformer KV-Cache

ACE (Adaptive Collapse Eviction) is a KV-cache eviction policy for long-context Transformers.
It retains tokens based on viability score v = w²/(t+ε), where:
    - w is learned token utility (predicted by a tiny MLP)
    - t is token age (decoding steps since insertion)

This package provides:
    - ACEKVCache: Main cache with min-heap eviction
    - UtilityMLP: MLP for predicting token utility w
    - ACEConfig: Configuration for all hyperparameters
    - train_utility_mlp: Training function for the MLP
    - load_utility_mlp: Load pretrained MLP from checkpoint
"""

# Version following paper date (May 2026)
__version__ = "1.0.0"
__author__ = "Stanislav Usychenko"
__date__ = "May 2026"

# Core components
from .cache import ACEKVCache, KVEntry
from .utility_mlp import UtilityMLP, SimpleUtilityMLP
from .config import (
    ACEConfig,
    get_llama2_7b_config,
    get_llama2_13b_config,
    get_llama2_70b_config,
    get_ablation_config,
)

# Training utilities
from .trainer import (
    train_utility_mlp,
    load_utility_mlp,
    compute_cumulative_attention,
    prepare_training_data,
)

# Public API
__all__ = [
    # Core classes
    "ACEKVCache",
    "KVEntry",
    "UtilityMLP",
    "SimpleUtilityMLP",
    "ACEConfig",
    # Preset configs
    "get_llama2_7b_config",
    "get_llama2_13b_config",
    "get_llama2_70b_config",
    "get_ablation_config",
    # Training functions
    "train_utility_mlp",
    "load_utility_mlp",
    "compute_cumulative_attention",
    "prepare_training_data",
]