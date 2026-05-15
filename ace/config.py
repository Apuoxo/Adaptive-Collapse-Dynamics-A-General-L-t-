from dataclasses import dataclass


@dataclass
class ACEConfig:
    """Configuration for ACE KV-cache eviction."""
    
    # Cache budget (max tokens to keep)
    cache_budget: int = 1024
    
    # Epsilon for numerical stability
    epsilon: float = 1e-8
    
    # MLP architecture
    mlp_hidden_dim: int = 128
    mlp_num_layers: int = 2
    
    # Local context window for utility prediction
    local_context_len: int = 5
    
    # Model dimensions (for LLaMA-2 7B)
    hidden_size: int = 4096
    num_heads: int = 32
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 3
    train_chunks: int = 10000
    
    def __post_init__(self):
        assert self.cache_budget > 0
        assert self.epsilon > 0
        assert 0 < self.mlp_hidden_dim <= 1024