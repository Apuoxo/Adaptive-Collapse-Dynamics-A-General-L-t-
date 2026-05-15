import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UtilityMLP(nn.Module):
    """
    Two-layer MLP that predicts token utility w ∈ [0,1].
    
    Based on the paper: w = σ(W₂ · ReLU(W₁ · h + b₁) + b₂)
    
    Input features h concatenates:
        1. Normalized key vector (k/√d) - flattened across heads
        2. Positional encoding scalar (pos/10000)
        3. Local context mean (mean of keys from preceding L tokens)
    
    Total input dimension: (num_heads * head_dim) * 2 + 1
    For LLaMA-2 7B: num_heads=32, head_dim=128 → 32*128 = 4096 per key
    So input_dim = 4096*2 + 1 = 8193
    
    The MLP has ~5000 parameters as per the paper:
        hidden_dim = 128 is used
        params = 8193*128 + 128*1 + biases ≈ 1,048,704 + 128 + 129 ≈ 1.05M
        Wait, that's >5000. Let me recalc...
        
    Actually the paper says "~5000 parameters". That suggests a much smaller architecture.
    Possibly they use a smaller key representation (e.g., projected) or different numbers.
    For LLaMA-2 7B, 8193*128 = 1,048,704 which is ~1M params.
    
    Let's adjust to achieve ~5000 params:
        - Option A: Use a smaller hidden_dim (e.g., 4)
        - Option B: Project key to lower dimension first
        - Option C: The paper might mean 5000 for the entire system? Or use a different model.
    
    Following the paper's spirit but making it practical, we'll use:
        - Projection layer to reduce key dimension from 4096 to 64
        - Then hidden_dim = 32
        Total: (64*2+1)*32 + 32*1 ≈ 4128 + 32 = 4160 params
    
    Or we can keep it simple and note that for LLaMA-2 7B, the MLP is larger but still tiny
    compared to the base model (7B vs 1M).
    """
    
    def __init__(self, 
                 num_heads: int = 32, 
                 head_dim: int = 128,
                 projection_dim: int = 64,
                 hidden_dim: int = 32,
                 local_context_len: int = 5):
        """
        Args:
            num_heads: Number of attention heads
            head_dim: Dimension per head
            projection_dim: Dimension to project key vectors (reduces input size)
            hidden_dim: Hidden layer dimension (kept small for ~5k params)
            local_context_len: Number of preceding tokens for context (L)
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_dim = num_heads * head_dim  # 32*128=4096
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.local_context_len = local_context_len
        
        # Projection layers for key and context
        self.key_proj = nn.Linear(self.key_dim, projection_dim, bias=False)
        self.context_proj = nn.Linear(self.key_dim, projection_dim, bias=False)
        
        # Main MLP: input = key_proj + pos_enc(1) + context_proj
        # input_dim = projection_dim * 2 + 1
        input_dim = projection_dim * 2 + 1
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following the paper."""
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, 
                key_norm: torch.Tensor, 
                pos_enc: torch.Tensor, 
                context_mean: torch.Tensor) -> torch.Tensor:
        """
        Predict utility w for a token.
        
        Args:
            key_norm: Normalized key vector [batch, 1, num_heads * head_dim] or [batch, key_dim]
            pos_enc: Positional encoding [batch, 1, 1] or [batch, 1]
            context_mean: Mean of previous L keys [batch, 1, key_dim] or [batch, key_dim]
        
        Returns:
            w: Utility in [0, 1] [batch, 1, 1] or [batch, 1]
        """
        # Handle input shapes
        if key_norm.dim() == 2:
            key_norm = key_norm.unsqueeze(1)
        if pos_enc.dim() == 1:
            pos_enc = pos_enc.unsqueeze(0).unsqueeze(-1)
        elif pos_enc.dim() == 2:
            pos_enc = pos_enc.unsqueeze(-1)
        if context_mean.dim() == 2:
            context_mean = context_mean.unsqueeze(1)
        
        batch_size = key_norm.shape[0]
        
        # Project key and context to lower dimension
        key_flat = key_norm.reshape(batch_size, -1)  # [batch, key_dim]
        ctx_flat = context_mean.reshape(batch_size, -1)  # [batch, key_dim]
        
        key_proj = self.key_proj(key_flat)  # [batch, proj_dim]
        ctx_proj = self.context_proj(ctx_flat)  # [batch, proj_dim]
        
        # Concatenate features
        pos_flat = pos_enc.reshape(batch_size, 1)  # [batch, 1]
        features = torch.cat([key_proj, pos_flat, ctx_proj], dim=-1)  # [batch, 2*proj_dim+1]
        
        # MLP forward
        x = F.relu(self.fc1(features))
        w = torch.sigmoid(self.fc2(x))
        
        # Return with same batch dimension, add trailing dims if needed
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        
        return w
    
    def forward_batch(self, 
                      keys: torch.Tensor, 
                      positions: torch.Tensor, 
                      contexts: torch.Tensor) -> torch.Tensor:
        """
        Batch forward for training.
        
        Args:
            keys: [batch, seq_len, key_dim]
            positions: [batch, seq_len] or [batch, seq_len, 1]
            contexts: [batch, seq_len, key_dim]
        
        Returns:
            w: [batch, seq_len, 1] utilities in [0,1]
        """
        batch_size, seq_len = keys.shape[:2]
        
        # Reshape to [batch*seq_len, ...]
        keys_flat = keys.reshape(-1, self.key_dim)
        positions_flat = positions.reshape(-1, 1)
        contexts_flat = contexts.reshape(-1, self.key_dim)
        
        # Forward
        w_flat = self.forward(keys_flat, positions_flat, contexts_flat)
        
        # Reshape back
        return w_flat.reshape(batch_size, seq_len, 1)
    
    @property
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config(self) -> dict:
        """Return model configuration for saving."""
        return {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "local_context_len": self.local_context_len,
            "num_parameters": self.num_parameters,
        }


class SimpleUtilityMLP(nn.Module):
    """
    Even simpler MLP for quick experimentation or smaller models.
    
    This version uses the full key dimension without projection,
    but with a very small hidden dimension to keep parameters low.
    
    For LLaMA-2 7B: key_dim=4096, hidden_dim=4
        input_dim = 4096*2 + 1 = 8193
        params = 8193*4 + 4*1 ≈ 32776 (still >5000)
    
    This confirms that ~5000 params requires projection or a smaller base model.
    """
    
    def __init__(self, key_dim: int = 4096, hidden_dim: int = 4):
        super().__init__()
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        
        input_dim = key_dim * 2 + 1  # key_norm + pos_enc + context_mean
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, key_norm: torch.Tensor, pos_enc: torch.Tensor, 
                context_mean: torch.Tensor) -> torch.Tensor:
        # Flatten
        if key_norm.dim() == 3:
            key_norm = key_norm.reshape(key_norm.shape[0], -1)
        if context_mean.dim() == 3:
            context_mean = context_mean.reshape(context_mean.shape[0], -1)
        if pos_enc.dim() == 3:
            pos_enc = pos_enc.reshape(pos_enc.shape[0], -1)
        
        # Concatenate
        x = torch.cat([key_norm, pos_enc, context_mean], dim=-1)
        
        # MLP
        x = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(x))
        
        return w
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())