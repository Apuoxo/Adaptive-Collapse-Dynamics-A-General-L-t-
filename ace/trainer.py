import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple
import os

from .utility_mlp import UtilityMLP
from .config import ACEConfig


def compute_cumulative_attention(model, input_ids, attention_mask=None):
    """
    Compute ground-truth utility as cumulative attention weight (as in H2O).
    
    For each token, utility = average attention it receives from all subsequent tokens.
    This reflects how "important" the token is for generating future tokens.
    
    Args:
        model: Transformer model
        input_ids: [batch, seq_len] input token ids
    
    Returns:
        utilities: [batch, seq_len] cumulative attention scores
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass with output attentions
        outputs = model(
            input_ids, 
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # attentions: tuple of [batch, num_heads, seq_len, seq_len] for each layer
        attentions = outputs.attentions
        
        # Stack all layers: [num_layers, batch, num_heads, seq_len, seq_len]
        stacked = torch.stack(attentions)
        
        # Average over layers and heads: [batch, seq_len, seq_len]
        avg_attn = stacked.mean(dim=(0, 2))  # (num_layers, heads) -> average
        
        # For each target position (column), sum attention received from all sources (rows)
        # Cumulative attention for token j = sum_i attention[i, j] where i >= j (future tokens)
        seq_len = avg_attn.shape[-1]
        utilities = torch.zeros(avg_attn.shape[0], seq_len, device=avg_attn.device)
        
        for batch_idx in range(avg_attn.shape[0]):
            for j in range(seq_len):
                # Sum attention from all future tokens (positions >= j)
                # Note: attention[i, j] = how much token i attends to token j
                utilities[batch_idx, j] = avg_attn[batch_idx, j:, j].sum()
        
        # Normalize to [0, 1] range
        utilities = utilities / (utilities.max(dim=-1, keepdim=True)[0] + 1e-8)
        
    return utilities


def extract_features_for_token(
    model, 
    input_ids, 
    token_position: int,
    local_context_len: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features needed for utility prediction for a single token.
    
    Args:
        model: Transformer model
        input_ids: [1, seq_len] input tokens
        token_position: Position index of the token
        local_context_len: Number of preceding tokens for context (L)
    
    Returns:
        key_norm: Normalized key vector [num_heads * head_dim]
        pos_enc: Positional encoding scalar
        context_mean: Mean of previous L keys [num_heads * head_dim]
    """
    with torch.no_grad():
        # Get hidden states from the model
        outputs = model(input_ids, output_hidden_states=True)
        
        # Last layer hidden states: [1, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]
        
        # Get the key projection (simplified: use hidden state as key)
        # In a real implementation, you'd extract actual K from attention layer
        # Here we approximate with hidden state
        key = hidden_states[0, token_position]  # [hidden_size]
        
        # Normalize: k / √d
        d = key.shape[0]
        key_norm = key / (d ** 0.5)
        
        # Positional encoding (sinusoidal, simplified)
        pos_enc = torch.tensor(token_position / 10000.0)
        
        # Context: mean of previous L keys
        start = max(0, token_position - local_context_len)
        if token_position > 0:
            context_keys = hidden_states[0, start:token_position]  # [L, hidden_size]
            context_mean = context_keys.mean(dim=0)
        else:
            context_mean = torch.zeros_like(key)
        
    return key_norm, pos_enc, context_mean


def prepare_training_data(
    model,
    tokenizer,
    config: ACEConfig,
    device: torch.device,
    num_chunks: int = 10000,
    chunk_length: int = 512
) -> TensorDataset:
    """
    Prepare training data for utility MLP.
    
    Loads chunks from C4 dataset, computes ground-truth utilities via
    cumulative attention, and extracts features.
    
    Args:
        model: Base transformer model
        tokenizer: Tokenizer for the model
        config: ACE configuration
        device: Device to run computations on
        num_chunks: Number of text chunks to process
        chunk_length: Length of each chunk in tokens
    
    Returns:
        TensorDataset with (keys, positions, contexts, utilities)
    """
    print(f"Loading C4 dataset and preparing {num_chunks} chunks...")
    
    # Load C4 dataset (streaming to avoid downloading everything)
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    
    all_keys = []
    all_positions = []
    all_contexts = []
    all_utilities = []
    
    chunk_count = 0
    pbar = tqdm(total=num_chunks, desc="Processing chunks")
    
    for example in dataset:
        if chunk_count >= num_chunks:
            break
        
        # Tokenize
        text = example["text"]
        tokens = tokenizer(
            text, 
            truncation=True, 
            max_length=chunk_length,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Skip very short sequences
        if input_ids.shape[1] < 10:
            continue
        
        try:
            # Compute ground-truth utilities (cumulative attention)
            utilities = compute_cumulative_attention(model, input_ids, attention_mask)
            utilities = utilities[0].cpu()  # [seq_len]
            
            # Extract features for each token position
            seq_len = input_ids.shape[1]
            for pos in range(seq_len):
                key_norm, pos_enc, context_mean = extract_features_for_token(
                    model, input_ids, pos, config.local_context_len
                )
                
                all_keys.append(key_norm.cpu())
                all_positions.append(pos_enc.cpu())
                all_contexts.append(context_mean.cpu())
                all_utilities.append(utilities[pos])
            
            chunk_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"Warning: Failed to process chunk {chunk_count}: {e}")
            continue
    
    pbar.close()
    
    # Convert to tensors
    print(f"Collected {len(all_keys)} token samples")
    
    keys_tensor = torch.stack(all_keys)           # [N, key_dim]
    positions_tensor = torch.stack(all_positions).unsqueeze(-1)  # [N, 1]
    contexts_tensor = torch.stack(all_contexts)   # [N, key_dim]
    utilities_tensor = torch.stack(all_utilities).unsqueeze(-1)  # [N, 1]
    
    # Normalize features
    keys_tensor = keys_tensor / (keys_tensor.norm(dim=-1, keepdim=True) + 1e-8)
    contexts_tensor = contexts_tensor / (contexts_tensor.norm(dim=-1, keepdim=True) + 1e-8)
    
    return TensorDataset(keys_tensor, positions_tensor, contexts_tensor, utilities_tensor)


def train_utility_mlp(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    config: Optional[ACEConfig] = None,
    save_path: str = "checkpoints/utility_mlp.pt",
    device: Optional[torch.device] = None
) -> UtilityMLP:
    """
    Train the utility MLP using cumulative attention as ground truth.
    
    Args:
        model_name: Name or path of the base model
        config: ACE configuration
        save_path: Where to save the trained MLP
        device: Device to train on
    
    Returns:
        Trained UtilityMLP model
    """
    if config is None:
        config = ACEConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"MLP parameters target: ~{config.mlp_hidden_dim * 1000}")
    
    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True  # Need attentions for ground truth
    )
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training data
    train_dataset = prepare_training_data(
        model, tokenizer, config, device,
        num_chunks=config.train_chunks,
        chunk_length=512
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    # Initialize MLP
    # Get key dimension from the model
    hidden_size = config.hidden_size
    num_heads = config.num_heads
    head_dim = hidden_size // num_heads  # 4096 // 32 = 128
    
    mlp = UtilityMLP(
        num_heads=num_heads,
        head_dim=head_dim,
        projection_dim=64,
        hidden_dim=config.mlp_hidden_dim,
        local_context_len=config.local_context_len
    )
    mlp.to(device)
    
    print(f"MLP has {mlp.num_parameters:,} parameters")
    
    # Training setup
    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    mlp.train()
    
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        
        for batch_keys, batch_pos, batch_ctx, batch_util in pbar:
            batch_keys = batch_keys.to(device)
            batch_pos = batch_pos.to(device)
            batch_ctx = batch_ctx.to(device)
            batch_util = batch_util.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = mlp(batch_keys, batch_pos, batch_ctx)
            loss = criterion(pred, batch_util)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")
    
    # Save checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'config': mlp.get_config(),
        'epoch': config.num_epochs,
        'loss': avg_loss
    }, save_path)
    
    print(f"Model saved to {save_path}")
    
    return mlp


def load_utility_mlp(
    checkpoint_path: str = "checkpoints/utility_mlp.pt",
    device: Optional[torch.device] = None
) -> UtilityMLP:
    """
    Load a pretrained utility MLP from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
    
    Returns:
        Loaded UtilityMLP model in eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']
    
    mlp = UtilityMLP(
        num_heads=config_dict.get('num_heads', 32),
        head_dim=config_dict.get('head_dim', 128),
        projection_dim=config_dict.get('projection_dim', 64),
        hidden_dim=config_dict.get('hidden_dim', 32),
        local_context_len=config_dict.get('local_context_len', 5)
    )
    
    mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.to(device)
    mlp.eval()
    
    print(f"Loaded MLP from {checkpoint_path} ({mlp.num_parameters:,} params)")
    
    return mlp