from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ACEConfig:
    """
    Configuration for Adaptive Collapse Eviction (ACE).
    
    Contains all hyperparameters for the KV-cache eviction policy
    based on the L²/t principle.
    
    Usage:
        config = ACEConfig(cache_budget=1024)
        cache = ACEKVCache(config)
    """
    
    # ========== Cache Settings ==========
    cache_budget: int = 1024
    """Maximum number of tokens to keep in the KV cache."""
    
    epsilon: float = 1e-8
    """Small constant for numerical stability in viability score v = w²/(t+ε)."""
    
    # ========== MLP Architecture ==========
    mlp_hidden_dim: int = 32
    """Hidden dimension of the utility prediction MLP."""
    
    projection_dim: int = 64
    """Dimension to project key vectors before MLP (reduces input size)."""
    
    local_context_len: int = 5
    """Number of preceding tokens (L) for local context mean."""
    
    # ========== Model Dimensions (for LLaMA-2 7B) ==========
    hidden_size: int = 4096
    """Hidden size of the transformer model."""
    
    num_heads: int = 32
    """Number of attention heads."""
    
    num_layers: int = 32
    """Number of transformer layers."""
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_heads
    
    @property
    def key_dim(self) -> int:
        """Total key dimension (num_heads * head_dim)."""
        return self.num_heads * self.head_dim
    
    # ========== Training Settings ==========
    learning_rate: float = 1e-3
    """Learning rate for training the utility MLP."""
    
    batch_size: int = 32
    """Batch size for MLP training."""
    
    num_epochs: int = 3
    """Number of training epochs."""
    
    train_chunks: int = 10000
    """Number of text chunks from C4 to use for training."""
    
    chunk_length: int = 512
    """Maximum length of each text chunk in tokens."""
    
    # ========== Inference Settings ==========
    min_viability_threshold: Optional[float] = None
    """
    Optional minimum viability threshold. Tokens with v < threshold are evicted
    immediately, regardless of cache budget. If None, no threshold is applied.
    """
    
    use_heuristic_when_no_mlp: bool = True
    """If True and MLP is not loaded, use heuristic utility (e.g., recency)."""
    
    # ========== Advanced ==========
    age_normalization: str = "linear"  # "linear", "sqrt", "square"
    """
    Type of age decay in viability score.
    - "linear": v = w²/(t+ε)  (default, as per paper)
    - "sqrt":   v = w²/√(t+ε) (slower decay)
    - "square": v = w²/(t+ε)² (faster decay, for ablation)
    """
    
    utility_power: int = 2
    """Power for utility in viability score. Default 2 gives w²/t."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.cache_budget > 0, "cache_budget must be positive"
        assert self.epsilon > 0, "epsilon must be positive"
        assert 0 < self.mlp_hidden_dim <= 1024, "mlp_hidden_dim must be in (0, 1024]"
        assert 0 < self.projection_dim <= self.key_dim, "projection_dim too large"
        assert self.local_context_len >= 0, "local_context_len must be >= 0"
        assert 0 < self.learning_rate <= 1.0, "learning_rate must be in (0, 1]"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.age_normalization in ["linear", "sqrt", "square"], \
            f"age_normalization must be one of linear, sqrt, square, got {self.age_normalization}"
        assert 1 <= self.utility_power <= 4, "utility_power must be between 1 and 4"
    
    def get_viability(self, utility: float, age: int) -> float:
        """
        Compute viability score v = w^p / f(t+ε) where f depends on age_normalization.
        
        Args:
            utility: Token utility w in [0, 1]
            age: Token age (decoding steps since insertion)
        
        Returns:
            Viability score v
        """
        w_power = utility ** self.utility_power
        t_eps = age + self.epsilon
        
        if self.age_normalization == "linear":
            return w_power / t_eps
        elif self.age_normalization == "sqrt":
            return w_power / (t_eps ** 0.5)
        else:  # square
            return w_power / (t_eps ** 2)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "cache_budget": self.cache_budget,
            "epsilon": self.epsilon,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "projection_dim": self.projection_dim,
            "local_context_len": self.local_context_len,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "train_chunks": self.train_chunks,
            "chunk_length": self.chunk_length,
            "age_normalization": self.age_normalization,
            "utility_power": self.utility_power,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ACEConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    def __repr__(self) -> str:
        lines = [f"ACEConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"    {key}={value},")
        lines.append(")")
        return "\n".join(lines)


# ========== Preset Configurations ==========

def get_llama2_7b_config(cache_budget: int = 1024) -> ACEConfig:
    """Default configuration for LLaMA-2 7B."""
    return ACEConfig(
        cache_budget=cache_budget,
        hidden_size=4096,
        num_heads=32,
        num_layers=32,
        mlp_hidden_dim=32,
        projection_dim=64,
    )


def get_llama2_13b_config(cache_budget: int = 1024) -> ACEConfig:
    """Configuration for LLaMA-2 13B."""
    return ACEConfig(
        cache_budget=cache_budget,
        hidden_size=5120,
        num_heads=40,
        num_layers=40,
        mlp_hidden_dim=48,
        projection_dim=80,
    )


def get_llama2_70b_config(cache_budget: int = 1024) -> ACEConfig:
    """Configuration for LLaMA-2 70B."""
    return ACEConfig(
        cache_budget=cache_budget,
        hidden_size=8192,
        num_heads=64,
        num_layers=80,
        mlp_hidden_dim=64,
        projection_dim=128,
    )


def get_ablation_config(variant: str) -> ACEConfig:
    """
    Get config for ablation studies.
    
    Variants:
        - "no_mlp": Random utility (w = 0.5)
        - "linear_w": Use w instead of w² (utility_power=1)
        - "square_age": Use 1/t² instead of 1/t (age_normalization="square")
        - "sqrt_age": Use 1/√t instead of 1/t (age_normalization="sqrt")
    """
    config = ACEConfig()
    
    if variant == "no_mlp":
        # Random utility (no MLP training needed)
        pass
    elif variant == "linear_w":
        config.utility_power = 1
    elif variant == "square_age":
        config.age_normalization = "square"
    elif variant == "sqrt_age":
        config.age_normalization = "sqrt"
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    
    return config