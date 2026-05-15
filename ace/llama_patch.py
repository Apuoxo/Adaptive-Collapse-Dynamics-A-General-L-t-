"""
Patched LLaMA attention with ACE KV-cache integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .cache import ACEKVCache
from .config import ACEConfig


class ACEAttention(LlamaAttention):
    """
    LlamaAttention with ACE KV-cache eviction.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int, ace_config: ACEConfig, utility_predictor=None):
        super().__init__(config, layer_idx)
        self.ace_config = ace_config
        self.utility_predictor = utility_predictor
        self.layer_idx = layer_idx
        self.ace_cache = None
        self._is_prefill = True  # Track prefill vs generation phase
    
    def set_ace_cache(self, ace_cache: ACEKVCache):
        self.ace_cache = ace_cache
    
    def _get_token_utility(self, key_states: torch.Tensor, position_ids: torch.Tensor) -> float:
        """Predict utility w for a single token."""
        if self.utility_predictor is None:
            return 0.5
        
        # key_states: [1, num_heads, 1, head_dim]
        head_dim = key_states.shape[-1]
        key_norm = key_states / (head_dim ** 0.5)
        key_flat = key_norm.reshape(1, -1)  # [1, num_heads * head_dim]
        
        pos_enc = torch.tensor([[position_ids[0, -1].item() / 10000.0]], 
                               device=key_states.device, dtype=key_states.dtype)
        
        # Local context (simplified: use zeros for now)
        context_mean = torch.zeros_like(key_flat)
        
        with torch.no_grad():
            w = self.utility_predictor(key_flat, pos_enc, context_mean)
        
        return float(w.squeeze())
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # ========== ACE Cache ==========
        if use_cache and self.ace_cache is not None:
            # During prefill, add all tokens at once
            if self._is_prefill or q_len > 1:
                # Prefill phase: add all tokens to cache without eviction logic
                # (simplified: use standard KV cache for prefill)
                self._is_prefill = False
            else:
                # Generation phase: single new token
                # Get token ID
                if cache_position is not None and len(cache_position) > 0:
                    token_id = cache_position[-1].item()
                else:
                    token_id = self.ace_cache.current_step
                
                # Predict utility
                utility = self._get_token_utility(key_states, position_ids)
                
                # Append to ACE cache
                self.ace_cache.append(
                    keys=key_states.squeeze(0),  # [num_heads, head_dim]
                    values=value_states.squeeze(0),
                    token_id=token_id,
                    position=position_ids[0, -1].item(),
                    layer_idx=self.layer_idx,
                    utility=utility,  # Need to update cache.py to accept utility
                )
                
                # Update ages of all cached tokens
                self.ace_cache.update_ages()
        
        # Get current cache content (if using ACE)
        if use_cache and self.ace_cache is not None and len(self.ace_cache) > 0:
            cached_keys, cached_values = self.ace_cache.get_kv_for_layer(self.layer_idx)
            # Reshape to [1, num_heads, seq_len, head_dim]
            if cached_keys.numel() > 0:
                cached_keys = cached_keys.unsqueeze(0)  # [1, cache_len, num_heads, head_dim]
                cached_keys = cached_keys.permute(0, 2, 1, 3)  # [1, num_heads, cache_len, head_dim]
                cached_values = cached_values.unsqueeze(0).permute(0, 2, 1, 3)
                
                # Concatenate with current? Or replace? ACE cache already includes current.
                # For generation, cache already has all previous + current
                key_states = cached_keys
                value_states = cached_values
        elif past_key_value is not None:
            # Fallback to standard cache
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # Repeat KV for grouped-query attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        # Return format compatible with transformers
        return attn_output, None, (key_states, value_states)


def patch_llama_with_ace(
    model,
    ace_config: ACEConfig,
    utility_predictor=None,
    cache_budget: int = 1024,
) -> Tuple[nn.Module, ACEKVCache]:
    """Patch LLaMA model to use ACE KV-cache."""
    
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
    
    # Replace attention layers
    for layer_idx, layer in enumerate(model.model.layers):
        new_attn = ACEAttention(
            config=model.config,
            layer_idx=layer_idx,
            ace_config=ace_config,
            utility_predictor=utility_predictor,
        )
        
        # Copy weights from original
        original = layer.self_attn
        new_attn.q_proj = original.q_proj
        new_attn.k_proj = original.k_proj
        new_attn.v_proj = original.v_proj
        new_attn.o_proj = original.o_proj
        new_attn.rotary_emb = original.rotary_emb
        new_attn.num_key_value_groups = original.num_key_value_groups
        
        new_attn.set_ace_cache(ace_cache)
        layer.self_attn = new_attn
    
    return model, ace_cache