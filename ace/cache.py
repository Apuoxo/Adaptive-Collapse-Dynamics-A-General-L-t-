import heapq
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class KVEntry:
    """
    Entry in KV cache with viability score.
    
    Viability: v = w² / (t + ε)
    where:
        w = utility (learned)
        t = age (decoding steps since insertion)
        ε = epsilon (small constant)
    """
    
    __slots__ = ("key", "value", "utility", "age", "viability", "token_id", "layer_idx")
    
    def __init__(self, key: torch.Tensor, value: torch.Tensor, 
                 utility: float, token_id: int, layer_idx: int = 0, age: int = 0):
        self.key = key          # [head_dim] or [num_heads, head_dim]
        self.value = value      # [head_dim] or [num_heads, head_dim]
        self.utility = utility
        self.age = age
        self.token_id = token_id
        self.layer_idx = layer_idx
        self.viability = (utility ** 2) / (age + 1e-8)
    
    def update_age(self, new_age: int) -> None:
        """Update age and recompute viability."""
        self.age = new_age
        self.viability = (self.utility ** 2) / (new_age + 1e-8)
    
    def __lt__(self, other: 'KVEntry') -> bool:
        """
        For min-heap: lower viability = higher priority for eviction.
        Python's heapq is min-heap, so we want smallest viability at root.
        """
        return self.viability < other.viability
    
    def __repr__(self) -> str:
        return f"KVEntry(id={self.token_id}, w={self.utility:.3f}, t={self.age}, v={self.viability:.4f})"


class ACEKVCache:
    """
    KV-cache with Adaptive Collapse Eviction.
    
    Maintains a fixed-size cache of key-value pairs.
    When full, evicts token with smallest viability v = w²/(t+ε).
    
    Uses a min-heap for O(log M) eviction.
    """
    
    def __init__(self, 
                 cache_budget: int = 1024,
                 epsilon: float = 1e-8,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 head_dim: int = 128):
        """
        Args:
            cache_budget: Maximum number of tokens to keep in cache
            epsilon: Small constant for numerical stability
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
        """
        self.cache_budget = cache_budget
        self.epsilon = epsilon
        
        # Cache storage: separate for each layer
        # Structure: [layer_idx] -> min-heap of KVEntry
        self.caches: List[List[KVEntry]] = [[] for _ in range(num_layers)]
        
        # Quick lookup: token_id -> list of entries (one per layer)
        self.token_to_entries: Dict[int, List[Optional[KVEntry]]] = {}
        
        # Current age counter
        self.current_step = 0
        
        # Model dimensions
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Optional MLP for utility prediction
        self.utility_predictor = None
    
    def set_utility_predictor(self, predictor: nn.Module) -> None:
        """Set the MLP that predicts token utility w."""
        self.utility_predictor = predictor
        self.utility_predictor.eval()
    
    def _get_utility(self, key: torch.Tensor, position: int, 
                     context_keys: Optional[torch.Tensor] = None) -> float:
        """
        Predict utility w for a new token using the MLP.
        
        Args:
            key: Key vector for this token [num_heads, head_dim]
            position: Position in sequence
            context_keys: Keys of previous L tokens for context [L, num_heads, head_dim]
        
        Returns:
            Utility w in [0, 1]
        """
        if self.utility_predictor is None:
            # Fallback: random utility (for ablation)
            return 0.5
        
        with torch.no_grad():
            # Normalize key: k / √d
            key_norm = key / (key.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Positional encoding (scalar)
            pos_enc = torch.tensor([[position / 10000.0]], dtype=key.dtype, device=key.device)
            
            # Local context mean
            if context_keys is not None and context_keys.numel() > 0:
                context_mean = context_keys.mean(dim=0, keepdim=True)  # [1, num_heads, head_dim]
                context_mean = context_mean.reshape(1, -1)  # flatten
            else:
                context_mean = torch.zeros(1, self.num_heads * self.head_dim, 
                                          dtype=key.dtype, device=key.device)
            
            # Flatten key
            key_flat = key_norm.reshape(1, -1)  # [1, num_heads * head_dim]
            
            # Concatenate and predict
            features = torch.cat([key_flat, pos_enc, context_mean], dim=-1)
            w = self.utility_predictor(features)
            return float(w.squeeze())
    
    def append(self, 
               keys: torch.Tensor, 
               values: torch.Tensor, 
               token_id: int,
               position: int,
               context_keys_list: Optional[List[torch.Tensor]] = None) -> None:
        """
        Append a new token's keys and values to the cache.
        
        Args:
            keys: [num_layers, num_heads, head_dim] or list of tensors
            values: [num_layers, num_heads, head_dim]
            token_id: Unique ID for this token
            position: Position in sequence (for positional encoding)
            context_keys_list: List of context keys per layer
        """
        # Ensure we have per-layer lists
        if context_keys_list is None:
            context_keys_list = [None] * self.num_layers
        
        # Track entries for this token across layers
        self.token_to_entries[token_id] = [None] * self.num_layers
        
        for layer_idx in range(self.num_layers):
            key = keys[layer_idx] if isinstance(keys, list) else keys[layer_idx]
            value = values[layer_idx] if isinstance(values, list) else values[layer_idx]
            
            # Get utility (use layer 0 as representative, or average across layers)
            context = context_keys_list[layer_idx]
            utility = self._get_utility(key, position, context)
            
            entry = KVEntry(
                key=key.clone(),
                value=value.clone(),
                utility=utility,
                token_id=token_id,
                layer_idx=layer_idx,
                age=0
            )
            
            cache = self.caches[layer_idx]
            
            if len(cache) < self.cache_budget:
                heapq.heappush(cache, entry)
            else:
                # Evict the smallest viability
                if entry.viability > cache[0].viability:
                    evicted = heapq.heapreplace(cache, entry)
                    # Remove from token mapping (only if all layers evicted)
                    self._cleanup_evicted_token(evicted.token_id, layer_idx)
                # else: new token has lower viability, discard it (don't add to cache)
            
            self.token_to_entries[token_id][layer_idx] = entry
    
    def _cleanup_evicted_token(self, token_id: int, layer_idx: int) -> None:
        """Clean up token mapping when entry is evicted from a layer."""
        if token_id in self.token_to_entries:
            self.token_to_entries[token_id][layer_idx] = None
            
            # If all layers have no entry for this token, remove the mapping
            if all(e is None for e in self.token_to_entries[token_id]):
                del self.token_to_entries[token_id]
    
    def update_ages(self) -> None:
        """Increment age of all cached tokens and recompute viability."""
        self.current_step += 1
        
        for layer_idx, cache in enumerate(self.caches):
            new_cache = []
            for entry in cache:
                # New age = current_step - (step when inserted)
                # We store insertion step separately; simplified: age += 1
                entry.update_age(entry.age + 1)
                new_cache.append(entry)
            heapq.heapify(new_cache)
            self.caches[layer_idx] = new_cache
    
    def get_kv_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get concatenated keys and values for a specific layer.
        
        Returns:
            keys: [cache_size, num_heads, head_dim] or [cache_size, head_dim]
            values: [cache_size, num_heads, head_dim]
        """
        cache = self.caches[layer_idx]
        if not cache:
            return torch.empty(0), torch.empty(0)
        
        # Sort by token_id to maintain original order? 
        # For attention, order doesn't matter as long as positions are correct.
        # We return in heap order (any order is fine).
        keys = torch.stack([e.key for e in cache])
        values = torch.stack([e.value for e in cache])
        return keys, values
    
    def get_all_kv(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get keys and values for all layers.
        
        Returns:
            List of (keys, values) per layer, compatible with transformers' past_key_values.
        """
        return [self.get_kv_for_layer(i) for i in range(self.num_layers)]
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.caches = [[] for _ in range(self.num_layers)]
        self.token_to_entries.clear()
        self.current_step = 0
    
    def __len__(self) -> int:
        """Return number of tokens in cache (assuming all layers have same size)."""
        return len(self.caches[0]) if self.caches else 0
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self),
            "budget": self.cache_budget,
            "step": self.current_step,
            "num_tokens_tracked": len(self.token_to_entries),
            "layers": self.num_layers,
        }