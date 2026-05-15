import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from .utility_mlp import UtilityMLP
from .config import ACEConfig


def compute_cumulative_attention(model, tokens):
    """Ground-truth utility = cumulative attention weight (as in H2O)."""
    with torch.no_grad():
        outputs = model(tokens, output_attentions=True)
        attentions = outputs.attentions  # tuple of [batch, heads, seq, seq]
        
        # Average over layers and heads
        stacked = torch.stack(attentions)
        avg_attn = stacked.mean(dim=(0, 1, -1))  # [batch, seq]
        return avg_attn.squeeze()


def prepare_training_data(model, tokenizer, config: ACEConfig, device):
    """Prepare training data: tokens + ground-truth utilities."""
    dataset = load_dataset("c4", "en", split="train", streaming=True)
    
    all_keys = []
    all_positions = []
    all_contexts = []
    all_utilities = []
    
    chunks = []
    for i, example in enumerate(dataset):
        if i >= config.train_chunks * 10:
            break
        tokens = tokenizer(example["text"], truncation=True, 
                          max_length=512, return_tensors="pt")
        chunks.append(tokens)
        if len(chunks) >= config.batch_size:
            break
    
    for tokens in tqdm(chunks[:config.train_chunks], desc="Computing ground truth"):
        tokens = {k: v.to(device) for k, v in tokens.items()}
        utilities = compute_cumulative_attention(model, tokens["input_ids"])
        
        # Extract features
        with torch.no_grad():
            outputs = model(tokens["input_ids"], output_hidden_states=True)
            keys = outputs.hidden_states[-1]  # last layer hidden states
        
        seq_len = keys.shape[1]
        for pos in range(seq_len):
            key_norm = keys[0, pos] / (keys[0, pos].norm() + 1e-8)
            pos_enc = torch.tensor([pos / 10000])
            start = max(0, pos - config.local_context_len)
            context_mean = keys[0, start:pos].mean(dim=0) if pos > 0 else torch.zeros_like(key_norm)
            
            all_keys.append(key_norm.cpu())
            all_positions.append(pos_enc)
            all_contexts.append(context_mean.cpu())
            all_utilities.append(utilities[0, pos].cpu())
    
    return (torch.stack(all_keys), torch.stack(all_positions).unsqueeze(-1),
            torch.stack(all_contexts), torch.tensor(all_utilities).unsqueeze(-1))


def train_utility_mlp(model_name: str = "meta-llama/Llama-2-7b-hf",
                      config: ACEConfig = None) -> UtilityMLP:
    """Train the utility MLP using cumulative attention as ground truth."""
    if config is None:
        config = ACEConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare training data
    print("Preparing training data...")
    keys, positions, contexts, utilities = prepare_training_data(
        model, tokenizer, config, device
    )
    
    # Initialize MLP
    mlp = UtilityMLP(hidden_size=config.hidden_size, 
                     hidden_dim=config.mlp_hidden_dim)
    mlp.to(device)
    
    # Training
    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(keys, positions, contexts, utilities)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"Training MLP ({mlp.num_parameters} parameters)...")
    mlp.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_keys, batch_pos, batch_ctx, batch_util in pbar:
            batch_keys = batch_keys.to(device)
            batch_pos = batch_pos.to(device)
            batch_ctx = batch_ctx.to(device)
            batch_util = batch_util.to(device)
            
            optimizer.zero_grad()
            pred = mlp(batch_keys, batch_pos, batch_ctx)
            loss = criterion(pred, batch_util)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} average loss: {total_loss/len(dataloader):.6f}")
    
    # Save checkpoint
    torch.save(mlp.state_dict(), "checkpoints/utility_mlp.pt")
    print("Saved to checkpoints/utility_mlp.pt")
    
    return mlp