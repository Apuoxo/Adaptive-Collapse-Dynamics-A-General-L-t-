"""
Patched LLaMA attention with ACE KV-cache integration.

This module replaces the standard attention mechanism in LLaMA models
with a version that supports ACE (Adaptive Collapse Eviction) cache.

Usage:
    from ace.llama_patch import patch_llama_with_ace
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = patch_llama_with_ace(model, config, utility_predictor)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.cache_utils import Cache

from .cache import ACEKVCache
from .config import ACEConfig


class ACEAttention(LlamaAttention):
    """
    LlamaAttention with ACE KV-cache eviction.
    
    Overrides the forward method to use ACE cache instead of
    the standard transformers cache.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int, ace_config: ACEConfig, utility_predictor=None):
        super().__init__(config, layer_idx)
        self.ace_config = ace_config
        self.utility_predictor = utility_predictor
        self.layer_idx = layer_idx
        
        # Per-layer ACE cache (shared across layers via reference)
        self.ace_cache = None
    
    def set_ace_cache(self, ace_cache: ACEKVCache):
        """Set the shared ACE cache for this layer."""
        self.ace_cache = ace_cache
    
    def _get_token_utility(self, key_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict utility w for each token using the MLP.
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]
        
        Returns:
            utilities: [batch, seq_len, 1] in [0,1]
        """
        if self.utility_predictor is None:
            # Fallback: use attention entropy as proxy utility
            return torch.ones(key_states.shape[2], device=key_states.device) * 0.5
        
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Normalize keys: k / sqrt(d)
        key_norm = key_states / (head_dim ** 0.5)
        
        # Flatten across heads for MLP input
        key_flat = key_norm.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [B, L, num_heads*head_dim]
        
        # Positional encoding (normalized)
        pos_enc = (position_ids.unsqueeze(-1).float() / 10000.0)  # [B, L, 1]
        
        # Local context mean (preceding L tokens)
        local_context_len = self.ace_config.local_context_len
        context_means = []
        for i in range(seq_len):
            start = max(0, i - local_context_len)
            if start < i:
                ctx_mean = key_flat[:, start:i, :].mean(dim=1, keepdim=True)  # [B, 1, D]
            else:
                ctx_mean = torch.zeros_like(key_flat[:, i:i+1, :])
            context_means.append(ctx_mean)
        context_means = torch.cat(context_means, dim=1)  # [B, L, D]
        
        # Predict utilities
        utilities = []
        for b in range(batch_size):
            batch_utilities = []
            for t in range(seq_len):
                w = self.utility_predictor(
                    key_flat[b:b+1, t:t+1, :],
                    pos_enc[b:b+1, t:t+1, :],
                    context_means[b:b+1, t:t+1, :]
                )
                batch_utilities.append(w.squeeze().item())
            utilities.append(batch_utilities)
        
        return torch.tensor(utilities, device=key_states.device).unsqueeze(-1)  # [B, L, 1]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with ACE KV-cache.
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Query, Key, Value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embedding
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # ========== ACE Cache Integration ==========
        if use_cache and self.ace_cache is not None:
            # Get current token ID (assuming sequential generation)
            current_token_id = cache_position[-1].item() if cache_position is not None else 0
            
            # Predict utility for the new token
            utilities = self._get_token_utility(key_states, position_ids)
            token_utility = utilities[0, -1, 0].item()  # utility of the newest token
            
            # Store in ACE cache
            self.ace_cache.append(
                keys=key_states,  # [1, num_heads, 1, head_dim]
                values=value_states,
                token_id=current_token_id,
                position=position_ids[0, -1].item(),
                layer_idx=self.layer_idx,
                utility=token_utility,
            )
            
            # Update ages of all cached tokens
            self.ace_cache.update_ages()
            
            # Retrieve the filtered KV from ACE cache
            cached_keys, cached_values = self.ace_cache.get_kv_for_layer(self.layer_idx)
            
            # Combine with current key/value if not already in cache
            # (Current token is already added, so we just use the cache)
            key_states = cached_keys.unsqueeze(0).transpose(1, 2)  # [1, num_heads, cache_len, head_dim]
            value_states = cached_values.unsqueeze(0).transpose(1, 2)
            
            # Update past_key_value to match ACE cache (for compatibility)
            if past_key_value is not None:
                past_key_value.update(key_states, value_states, self.layer_idx)
        
        # ========== Standard attention computation ==========
        # Repeat KV for grouped-query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            causal_mask = None
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


class ACEStaticCache:
    """
    Static cache wrapper for ACE that mimics transformers' Cache interface.
    """
    
    def __init__(self, ace_cache: ACEKVCache):
        self.ace_cache = ace_cache
        self._seen_tokens = 0
    
    def update(self, key_states, value_states, layer_idx):
        """Update cache (compatibility with transformers)."""
        # ACE cache is updated separately
        self._seen_tokens += 1
        return self
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return len(self.ace_cache)
    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy tuple format."""
        kv_list = []
        for layer_idx in range(self.ace_cache.num_layers):
            keys, values = self.ace_cache.get_kv_for_layer(layer_idx)
            kv_list.append((keys, values))
        return tuple(kv_list)


def patch_llama_with_ace(
    model,
    ace_config: ACEConfig,
    utility_predictor=None,
    cache_budget: int = 1024,
) -> Tuple[nn.Module, ACEKVCache]:
    """
    Patch a LLaMA model to use ACE KV-cache.
    
    Args:
        model: Pretrained LLaMA model
        ace_config: ACE configuration
        utility_predictor: Trained UtilityMLP model
        cache_budget: Maximum number of tokens to keep in cache
    
    Returns:
        (patched_model, ace_cache)
    """
    # Create ACE cache
    ace_cache = ACEKVCache(
        cache_budget=cache_budget,
        epsilon=ace_config.epsilon,
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
    )
    
    if utility_predictor is not None:
        ace_cache.set_utility_predictor(utility_predictor)
    
    # Replace each attention layer with ACEAttention
    for layer_idx, layer in enumerate(model.model.layers):
        new_attn = ACEAttention(
            config=model.config,
            layer_idx=layer_idx,
            ace_config=ace_config,
            utility_predictor=utility_predictor,
        )
        # Copy weights from original attention
        new_attn.q_proj = layer.self_attn.q_proj
        new_attn.k_proj = layer.self_attn.k_proj
        new_attn.v_proj = layer.self_attn.v_proj
        new_attn.o_proj = layer.self_attn.o_proj
        new_attn.rotary_emb = layer.self_attn.rotary_emb
        new_attn.num_key_value_groups = layer.self_attn.num_key_value_groups
        
        # Set ACE cache reference
        new_attn.set_ace_cache(ace_cache)
        
        # Replace
        layer.self_attn = new_attn
    
    return model, ace_cache


def is_model_patched(model) -> bool:
    """Check if a model has been patched with ACE."""
    return hasattr(model.model.layers[0].self_attn, 'ace_cache')