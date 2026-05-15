import heapq
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class KVEntry:
    """
    Entry in KV cache with viability score.
    
    Viability: v = w² / (t + ε)
    where:
        w = utility (learned)
        t = age (decoding steps since insertion)
        ε = epsilon (small constant)
    """
    
    __slots__ = ("key", "value", "utility", "age", "viability", "token_id", "layer_idx", "position")
    
    def __init__(self, 
                 key: torch.Tensor, 
                 value: torch.Tensor, 
                 utility: float, 
                 token_id: int, 
                 layer_idx: int = 0, 
                 position: int = 0,
                 age: int = 0):
        """
        Args:
            key: Key tensor [num_heads, head_dim] or [head_dim]
            value: Value tensor [num_heads, head_dim] or [head_dim]
            utility: Token utility w in [0, 1]
            token_id: Unique identifier for this token
            layer_idx: Which transformer layer this entry belongs to
            position: Position in the sequence
            age: Number of decoding steps since insertion
        """
        self.key = key.clone() if isinstance(key, torch.Tensor) else key
        self.value = value.clone() if isinstance(value, torch.Tensor) else value
        self.utility = float(utility)
        self.token_id = token_id
        self.layer_idx = layer_idx
        self.position = position
        self.age = age
        self._update_viability()
    
    def _update_viability(self) -> None:
        """Recompute viability score based on current utility and age."""
        self.viability = (self.utility ** 2) / (self.age + 1e-8)
    
    def update_age(self, new_age: int) -> None:
        """Update age and recompute viability."""
        self.age = new_age
        self._update_viability()
    
    def __lt__(self, other: 'KVEntry') -> bool:
        """
        For min-heap: lower viability = higher priority for eviction.
        """
        return self.viability < other.viability
    
    def __repr__(self) -> str:
        return f"KVEntry(id={self.token_id}, pos={self.position}, w={self.utility:.3f}, t={self.age}, v={self.viability:.4f})"


class ACEKVCache:
    """
    KV-cache with Adaptive Collapse Eviction.
    
    Maintains a fixed-size cache of key-value pairs across all layers.
    When full, evicts token with smallest viability v = w²/(t+ε).
    
    Uses per-layer min-heaps for O(log M) eviction per layer.
    
    Example:
        >>> config = ACEConfig(cache_budget=1024)
        >>> cache = ACEKVCache(config)
        >>> cache.set_utility_predictor(mlp)
        >>> 
        >>> # During generation
        >>> for token in generate():
        ...     cache.append(keys, values, token_id, position, layer_idx, utility)
        ...     cache.update_ages()
        ...     keys, values = cache.get_kv_for_layer(layer_idx)
    """
    
    def __init__(self, 
                 cache_budget: int = 1024,
                 epsilon: float = 1e-8,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 head_dim: int = 128,
                 device: Optional[torch.device] = None):
        """
        Args:
            cache_budget: Maximum number of tokens to keep in cache per layer
            epsilon: Small constant for numerical stability
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each attention head
            device: Device to store tensors on
        """
        self.cache_budget = cache_budget
        self.epsilon = epsilon
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Per-layer cache: list of min-heaps (list of KVEntry)
        self.caches: List[List[KVEntry]] = [[] for _ in range(num_layers)]
        
        # Quick lookup: token_id -> list of entries (one per layer, or None if evicted)
        self.token_to_entries: Dict[int, List[Optional[KVEntry]]] = {}
        
        # Track token positions for ordering (optional)
        self.token_positions: Dict[int, int] = {}
        
        # Current age counter (number of generation steps)
        self.current_step = 0
        
        # Utility predictor (MLP)
        self.utility_predictor: Optional[nn.Module] = None
        
        # Statistics
        self.num_evictions = 0
        self.num_appends = 0
    
    def set_utility_predictor(self, predictor: nn.Module) -> None:
        """Set the MLP that predicts token utility w."""
        self.utility_predictor = predictor
        self.utility_predictor.eval()
    
    def _predict_utility(self, 
                         key: torch.Tensor, 
                         position: int, 
                         context_keys: Optional[torch.Tensor] = None) -> float:
        """
        Predict utility w for a new token using the MLP.
        
        Args:
            key: Key vector [num_heads, head_dim] or [head_dim]
            position: Position in sequence
            context_keys: Keys of previous L tokens [L, num_heads, head_dim]
        
        Returns:
            Utility w in [0, 1]
        """
        if self.utility_predictor is None:
            # Fallback: recency-based utility (older tokens have lower utility)
            return 0.5
        
        with torch.no_grad():
            # Ensure correct shape
            if key.dim() == 1:
                # [dim] -> [1, dim]
                key = key.unsqueeze(0)
            
            # Normalize key: k / √d
            d = key.shape[-1]
            key_norm = key / (d ** 0.5)
            
            # Flatten
            key_flat = key_norm.reshape(1, -1)
            
            # Positional encoding (scalar, normalized)
            pos_enc = torch.tensor([[position / 10000.0]], dtype=key.dtype, device=key.device)
            
            # Local context mean
            if context_keys is not None and context_keys.numel() > 0:
                if context_keys.dim() == 3:
                    # [L, num_heads, head_dim] -> [1, num_heads * head_dim]
                    context_mean = context_keys.mean(dim=0, keepdim=True).reshape(1, -1)
                else:
                    context_mean = context_keys.mean(dim=0, keepdim=True)
            else:
                context_mean = torch.zeros_like(key_flat)
            
            # Concatenate features
            features = torch.cat([key_flat, pos_enc, context_mean], dim=-1)
            
            # Predict
            w = self.utility_predictor(features)
            return float(w.squeeze())
    
    def append(self, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               token_id: int,
               position: int,
               layer_idx: int,
               utility: Optional[float] = None,
               context_keys: Optional[torch.Tensor] = None) -> bool:
        """
        Append a new token's key and value to the cache for a specific layer.
        
        Args:
            key: Key tensor [num_heads, head_dim] or [head_dim]
            value: Value tensor [num_heads, head_dim] or [head_dim]
            token_id: Unique ID for this token
            position: Position in sequence
            layer_idx: Which layer this belongs to (0..num_layers-1)
            utility: Pre-computed utility (if None, will predict)
            context_keys: Context keys for utility prediction
        
        Returns:
            True if token was added, False if it was discarded (low viability)
        """
        # Predict utility if not provided
        if utility is None:
            utility = self._predict_utility(key, position, context_keys)
        
        # Create entry
        entry = KVEntry(
            key=key,
            value=value,
            utility=utility,
            token_id=token_id,
            layer_idx=layer_idx,
            position=position,
            age=0
        )
        
        # Get the cache for this layer
        cache = self.caches[layer_idx]
        
        # Track token across layers
        if token_id not in self.token_to_entries:
            self.token_to_entries[token_id] = [None] * self.num_layers
        self.token_to_entries[token_id][layer_idx] = entry
        self.token_positions[token_id] = position
        
        self.num_appends += 1
        
        # Add to cache (with eviction if full)
        if len(cache) < self.cache_budget:
            heapq.heappush(cache, entry)
            return True
        else:
            # Cache is full - evict smallest viability if new token is better
            if entry.viability > cache[0].viability:
                evicted = heapq.heapreplace(cache, entry)
                self.num_evictions += 1
                self._cleanup_evicted_token(evicted.token_id, layer_idx)
                return True
            else:
                # New token has lower viability, don't add it
                self._cleanup_evicted_token(token_id, layer_idx)
                return False
    
    def append_batch(self,
                     keys: List[torch.Tensor],
                     values: List[torch.Tensor],
                     token_id: int,
                     position: int,
                     utilities: Optional[List[float]] = None) -> List[bool]:
        """
        Append a token to all layers at once.
        
        Args:
            keys: List of key tensors for each layer
            values: List of value tensors for each layer
            token_id: Unique ID for this token
            position: Position in sequence
            utilities: Optional list of pre-computed utilities per layer
        
        Returns:
            List of booleans indicating if added to each layer
        """
        results = []
        for layer_idx in range(self.num_layers):
            utility = utilities[layer_idx] if utilities and layer_idx < len(utilities) else None
            result = self.append(
                key=keys[layer_idx],
                value=values[layer_idx],
                token_id=token_id,
                position=position,
                layer_idx=layer_idx,
                utility=utility
            )
            results.append(result)
        return results
    
    def _cleanup_evicted_token(self, token_id: int, layer_idx: int) -> None:
        """Clean up token mapping when entry is evicted from a layer."""
        if token_id in self.token_to_entries:
            self.token_to_entries[token_id][layer_idx] = None
            
            # If all layers have no entry for this token, remove the mapping
            if all(e is None for e in self.token_to_entries[token_id]):
                del self.token_to_entries[token_id]
                if token_id in self.token_positions:
                    del self.token_positions[token_id]
    
    def update_ages(self, steps: int = 1) -> None:
        """
        Increment age of all cached tokens and recompute viability.
        
        Args:
            steps: Number of steps to increment (default 1)
        """
        self.current_step += steps
        
        for layer_idx, cache in enumerate(self.caches):
            new_cache = []
            for entry in cache:
                entry.update_age(entry.age + steps)
                new_cache.append(entry)
            # Re-heapify after updating all entries
            heapq.heapify(new_cache)
            self.caches[layer_idx] = new_cache
    
    def get_kv_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get concatenated keys and values for a specific layer.
        
        Returns:
            keys: [cache_size, num_heads, head_dim] or [cache_size, head_dim]
            values: [cache_size, num_heads, head_dim]
            Returns empty tensors if cache is empty.
        """
        cache = self.caches[layer_idx]
        if not cache:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        
        # Sort by position to maintain order (important for attention)
        sorted_entries = sorted(cache, key=lambda e: e.position)
        
        keys = torch.stack([e.key for e in sorted_entries])
        values = torch.stack([e.value for e in sorted_entries])
        
        return keys, values
    
    def get_kv_for_layer_as_past_key_value(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get KV in format expected by transformers' past_key_value.
        
        Returns:
            (keys, values) where keys/values are [1, num_heads, cache_size, head_dim]
        """
        keys, values = self.get_kv_for_layer(layer_idx)
        
        if keys.numel() == 0:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)
        
        # Reshape to [batch=1, num_heads, seq_len, head_dim]
        if keys.dim() == 2:
            # [seq_len, dim] -> [1, 1, seq_len, dim] (assume single head)
            keys = keys.unsqueeze(0).unsqueeze(0)
            values = values.unsqueeze(0).unsqueeze(0)
        elif keys.dim() == 3:
            # [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
            keys = keys.permute(1, 0, 2).unsqueeze(0)
            values = values.permute(1, 0, 2).unsqueeze(0)
        
        return keys, values
    
    def get_all_kv(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get keys and values for all layers.
        
        Returns:
            List of (keys, values) per layer
        """
        return [self.get_kv_for_layer(i) for i in range(self.num_layers)]
    
    def get_all_kv_as_past_key_values(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Get KV in format compatible with transformers' past_key_values.
        
        Returns:
            Tuple of ((keys, values), ...) for each layer
        """
        result = []
        for i in range(self.num_layers):
            keys, values = self.get_kv_for_layer_as_past_key_value(i)
            result.append((keys, values))
        return tuple(result)
    
    def get_token_utility(self, token_id: int) -> Optional[float]:
        """Get utility of a specific token (from layer 0)."""
        if token_id in self.token_to_entries:
            entry = self.token_to_entries[token_id][0]
            if entry:
                return entry.utility
        return None
    
    def get_cache_size(self) -> int:
        """Return number of tokens in cache (assuming all layers have same size)."""
        return len(self.caches[0]) if self.caches else 0
    
    def is_full(self) -> bool:
        """Check if cache has reached its budget."""
        return self.get_cache_size() >= self.cache_budget
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.caches = [[] for _ in range(self.num_layers)]
        self.token_to_entries.clear()
        self.token_positions.clear()
        self.current_step = 0
        self.num_evictions = 0
        self.num_appends = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.get_cache_size(),
            "budget": self.cache_budget,
            "step": self.current_step,
            "num_tokens_tracked": len(self.token_to_entries),
            "num_evictions": self.num_evictions,
            "num_appends": self.num_appends,
            "eviction_rate": self.num_evictions / max(1, self.num_appends),
            "layers": self.num_layers,
        }
    
    def __len__(self) -> int:
        return self.get_cache_size()
    
    def __repr__(self) -> str:
        return f"ACEKVCache(budget={self.cache_budget}, size={len(self)}, evictions={self.num_evictions})"


# Alias for backward compatibility
ACEKVCacheV2 = ACEKVCache