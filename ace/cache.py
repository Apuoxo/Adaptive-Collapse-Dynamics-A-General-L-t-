import heapq
from typing import List, Tuple, Optional
import torch


class KVEntry:
    """Entry in KV cache with viability score."""
    
    __slots__ = ("key", "value", "utility", "age", "viability", "token_id")
    
    def __init__(self, key: torch.Tensor, value: torch.Tensor, 
                 utility: float, token_id: int, age: int = 0):
        self.key = key
        self.value = value
        self.utility = utility
        self.age = age
        self.token_id = token_id
        self.viability = utility ** 2 / (age + 1e-8)
    
    def update_age(self, new_age: int) -> None:
        self.age = new_age
        self.viability = self.utility ** 2 / (new_age + 1e-8)
    
    def __lt__(self, other):
        """For min-heap: lower viability = higher priority for eviction."""
        return self.viability < other.viability


class ACEKVCache:
    """
    KV-cache with Adaptive Collapse Eviction.
    
    Keeps top-M tokens by viability score v = w²/(t+ε).
    """
    
    def __init__(self, model, config, utility_predictor: Optional[nn.Module] = None):
        self.model = model
        self.config = config
        self.utility_predictor = utility_predictor
        
        self.cache_budget = config.cache_budget
        self.epsilon = config.epsilon
        
        self.entries: List[KVEntry] = []  # heap (min by viability)
        self.entry_map: dict = {}  # token_id -> entry
        self.current_step = 0
    
    def get_utility(self, key: torch.Tensor, pos: int, 
                    context_keys: torch.Tensor) -> float:
        """Predict utility w for a new token."""
        if self.utility_predictor is None:
            return 0.5  # default fallback
        
        with torch.no_grad():
            key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
            pos_enc = torch.tensor([[[pos / 10000]]], dtype=key.dtype, device=key.device)
            context_mean = context_keys.mean(dim=-2, keepdim=True) if context_keys.numel() > 0 else torch.zeros_like(key_norm)
            
            w = self.utility_predictor(key_norm.unsqueeze(0), pos_enc, context_mean.unsqueeze(0))
            return float(w.squeeze())
    
    def append(self, key: torch.Tensor, value: torch.Tensor, 
               token_id: int, position: int, context_keys: torch.Tensor) -> None:
        """Append a new token to cache."""
        utility = self.get_utility(key, position, context_keys)
        entry = KVEntry(key, value, utility, token_id, age=0)
        
        if len(self.entries) < self.cache_budget:
            heapq.heappush(self.entries, entry)
            self.entry_map[token_id] = entry
        else:
            # Evict the smallest viability
            if entry.viability > self.entries[0].viability:
                evicted = heapq.heapreplace(self.entries, entry)
                del self.entry_map[evicted.token_id]
                self.entry_map[token_id] = entry
    
    def update_ages(self) -> None:
        """Increment age of all cached tokens and recompute viability."""
        self.current_step += 1
        new_entries = []
        
        for entry in self.entries:
            entry.update_age(self.current_step - entry.age)
            new_entries.append(entry)
        
        heapq.heapify(new_entries)
        self.entries = new_entries
    
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get concatenated keys and values from cache."""
        if not self.entries:
            return torch.empty(0), torch.empty(0)
        
        keys = torch.stack([e.key for e in self.entries])
        values = torch.stack([e.value for e in self.entries])
        return keys, values
    
    def clear(self) -> None:
        """Clear the cache."""
        self.entries.clear()
        self.entry_map.clear()
        self.current_step = 0
    
    def __len__(self) -> int:
        return len(self.entries)