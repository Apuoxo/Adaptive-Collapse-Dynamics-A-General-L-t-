#!/usr/bin/env python3
"""
Training script for the Utility MLP.

Trains a small MLP to predict token utility w based on key vectors,
positional encoding, and local context. Ground truth is cumulative attention.

Usage:
    python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf
    python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf --epochs 5 --batch-size 64
    python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf --cache-budget 2048 --output checkpoints/my_mlp.pt
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import ACEConfig, train_utility_mlp


def main():
    parser = argparse.ArgumentParser(
        description="Train Utility MLP for ACE KV-cache eviction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training on LLaMA-2 7B
  python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf
  
  # Training with custom settings
  python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf --epochs 5 --batch-size 64 --chunks 20000
  
  # Save to custom path
  python scripts/train_utility_mlp.py --model meta-llama/Llama-2-7b-hf --output checkpoints/custom_mlp.pt
        """
    )
    
    # Model arguments
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name or path (default: meta-llama/Llama-2-7b-hf)")
    
    # Cache arguments
    parser.add_argument("--cache-budget", type=int, default=1024,
                        help="KV-cache budget (default: 1024)")
    parser.add_argument("--epsilon", type=float, default=1e-8,
                        help="Epsilon for numerical stability (default: 1e-8)")
    
    # MLP architecture
    parser.add_argument("--mlp-hidden-dim", type=int, default=32,
                        help="Hidden dimension of MLP (default: 32)")
    parser.add_argument("--projection-dim", type=int, default=64,
                        help="Projection dimension for keys (default: 64)")
    parser.add_argument("--local-context-len", type=int, default=5,
                        help="Number of preceding tokens for context (default: 5)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--chunks", type=int, default=10000,
                        help="Number of C4 chunks for training (default: 10000)")
    parser.add_argument("--chunk-length", type=int, default=512,
                        help="Length of each chunk in tokens (default: 512)")
    
    # Model dimensions (for non-standard models)
    parser.add_argument("--hidden-size", type=int, default=4096,
                        help="Model hidden size (default: 4096 for LLaMA-2 7B)")
    parser.add_argument("--num-heads", type=int, default=32,
                        help="Number of attention heads (default: 32 for LLaMA-2 7B)")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of transformer layers (default: 32 for LLaMA-2 7B)")
    
    # Output
    parser.add_argument("--output", type=str, default="checkpoints/utility_mlp.pt",
                        help="Output path for trained MLP (default: checkpoints/utility_mlp.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detected if not specified.")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ACE: Utility MLP Training")
    print("=" * 70)
    print(f"Base model:      {args.model}")
    print(f"Output path:     {args.output}")
    print(f"Cache budget:    {args.cache_budget}")
    print(f"MLP hidden dim:  {args.mlp_hidden_dim}")
    print(f"Projection dim:  {args.projection_dim}")
    print(f"Local context L: {args.local_context_len}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print(f"Training chunks: {args.chunks}")
    print("=" * 70)
    print()
    
    # Create configuration
    config = ACEConfig(
        cache_budget=args.cache_budget,
        epsilon=args.epsilon,
        mlp_hidden_dim=args.mlp_hidden_dim,
        projection_dim=args.projection_dim,
        local_context_len=args.local_context_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        train_chunks=args.chunks,
        chunk_length=args.chunk_length,
    )
    
    # Set device
    if args.device:
        import torch
        device = torch.device(args.device)
    else:
        device = None  # Auto-detect
    
    print("Starting training...")
    print("(This may take ~2 hours on an A100 GPU)")
    print()
    
    # Train the MLP
    mlp = train_utility_mlp(
        model_name=args.model,
        config=config,
        save_path=args.output,
        device=device
    )
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"MLP saved to: {args.output}")
    print(f"Total parameters: {mlp.num_parameters:,}")
    print()
    print("You can now use this MLP with ACE:")
    print(f"  from ace import ACEKVCache, load_utility_mlp")
    print(f"  mlp = load_utility_mlp('{args.output}')")
    print(f"  cache = ACEKVCache(cache_budget={args.cache_budget})")
    print(f"  cache.set_utility_predictor(mlp)")


if __name__ == "__main__":
    main()