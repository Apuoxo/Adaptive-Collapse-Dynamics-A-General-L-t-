import torch
import torch.nn as nn
import torch.nn.functional as F


class UtilityMLP(nn.Module):
    """
    Two-layer MLP that predicts token utility w ∈ [0,1].
    
    Input features:
    - Normalized key vector (hidden_size)
    - Positional encoding scalar
    - Local context mean (hidden_size)
    
    Total input dim: hidden_size * 2 + 1
    """
    
    def __init__(self, hidden_size: int = 4096, hidden_dim: int = 128):
        super().__init__()
        
        # Input: key_norm (4096) + pos_enc (1) + local_context_mean (4096) = 8193
        input_dim = hidden_size * 2 + 1
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, key_norm: torch.Tensor, pos_enc: torch.Tensor, 
                local_context_mean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            key_norm: [batch, seq_len, hidden_size] normalized keys (k/√d)
            pos_enc: [batch, seq_len, 1] positional encoding
            local_context_mean: [batch, seq_len, hidden_size] mean of previous L keys
        
        Returns:
            w: [batch, seq_len, 1] utility in [0,1]
        """
        # Concatenate features
        x = torch.cat([key_norm, pos_enc, local_context_mean], dim=-1)
        
        # MLP forward
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())