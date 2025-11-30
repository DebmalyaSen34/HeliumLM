import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import math
import time
import copy
from tokenizers import Tokenizer

from src.model.slm import TinySLM
from src.data.chat_dataset import ChatDataset
from src.data.alpaca_dataset import AlpacaChatDataset
from train import EarlyStopping, get_lr, evaluate


CONFIG = {
    # System
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'num_workers': 0,      # Set to 0 for simpler debugging, 4 for speed
    
    # Model Architecture (Must match what we built)
    'vocab_size': 32000,
    'd_model': 256,
    'n_layers': 8,
    'n_head': 8,
    'n_kv_head': 2,        # GQA
    'window_size': 64,     # SWA
    'max_seq_len': 256,    # Short context for faster training
    'mlp_ratio': 2.5,
    
    # Training (The Optimizer Math)
    'batch_size': 4,       # Micro-batch (fits in memory)
    'accum_steps': 8,      # Virtual Batch Size = 4 * 8 = 32
    'learning_rate': 3e-5, # A slower LR for finetuning
    'max_epochs': 2,       # Just a few epochs for finetuning
    'patience': 3,
    'weight_decay': 0.01,  # AdamW regularization
    'grad_clip': 1.0,      # Prevents exploding gradients
    
    # Data Paths
    'pretrained_model_path': 'checkpoints/best_model_01.pt',
    'tokenizer_path': 'data/tokenizer/tiny_slm_tokenizer.json',
    'save_dir': 'checkpoints',
    
    # Logging
    'use_wandb': False,    # Set to True if you have an account
    'log_interval': 10     # Print every 10 steps
}

def train():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    
    # Load Data
    dataset = AlpacaChatDataset(tokenizer_path=CONFIG['tokenizer_path'], max_seq_len=CONFIG['max_seq_len'])
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    val_ds = ChatDataset(tokenizer_path=CONFIG['tokenizer_path'], max_seq_len=CONFIG['max_seq_len'])
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=True if device == 'cuda' else False
    )
    
    # Initialize Model
    model = TinySLM(config=CONFIG).to(device=device)
    
    print("Loading pretrained model from:", CONFIG['pretrained_model_path'])
    if CONFIG['pretrained_model_path'] and os.path.isfile(CONFIG['pretrained_model_path']):
        state_dict = torch.load(CONFIG['pretrained_model_path'], map_location=device)
        model.load_state_dict(state_dict=state_dict)
        print("Pretrained model loaded successfully.")
    else:
        raise FileNotFoundError(f"Pretrained model not found at {CONFIG['pretrained_model_path']}")
    
    # Optimizer - Low LR
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scaler = torch.cuda.amp.GradScaler('cuda') if device == 'cuda' else None
    
    early_stopper = EarlyStopping(patience=CONFIG['patience'])
    
    if CONFIG['use_wandb']:
        import wandb
        wandb.init(project="tiny_slm_training", config=CONFIG, name="trial-04-chat")
        wandb.watch(model)
    
    
    # Training Loop
    total_steps = len(loader) * CONFIG['max_epochs'] // CONFIG['accum_steps']
    warmup_iters = int(total_steps * 0.1)
    print(f"Total finetuning steps: {total_steps}")
    
    iter_num = 0
    running_loss = 0.0
    
    best_model_weights = copy.deepcopy(model.state_dict()) 
    
    model.train()
    
    for epoch in range(CONFIG['max_epochs']):
        print(f"Starting epoch {epoch + 1}/{CONFIG['max_epochs']}")
        t0 = time.time()
        
        for batch_idx, (X, Y) in enumerate(loader):
            
            lr = get_lr(iter_num, max_iters=total_steps, warmup_iters=warmup_iters, min_lr=3e-5, max_lr=CONFIG['learning_rate'])
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            X, Y = X.to(device), Y.to(device)
            
            # Forward
            if scaler:
                with torch.cuda.amp.autocast('cuda'):
                    logits, _ = model(X[:, :-1])
                    
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, CONFIG['vocab_size']),
                        Y[:, 1:].reshape(-1),
                        ignore_index=dataset.pad_token_id # Don't learn padding
                    )
            else:
                logits, _ = model(X[:, :-1])
                
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, CONFIG['vocab_size']),
                    Y[:, 1:].reshape(-1),
                    ignore_index=dataset.pad_token_id
                )
                
            loss = loss / CONFIG['accum_steps'] # Normalize loss for accumulation
            
            # Backward
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            running_loss += loss.item() * CONFIG['accum_steps'] # Denormalize for logging
            
            # Step
            if (batch_idx + 1)%CONFIG['accum_steps'] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                iter_num+=1
                
                if iter_num % CONFIG['log_interval'] == 0:
                    avg_loss = running_loss / (CONFIG['log_interval']*CONFIG['accum_steps'])
                    
                    perplexity = math.exp(avg_loss) if avg_loss < 20 else -1
                    
                    tokens_processed = (CONFIG['batch_size']*CONFIG['max_seq_len']*CONFIG['accum_steps']*CONFIG['log_interval'])
                    
                    dt = time.time() - t0
                    tokens_per_sec = tokens_processed / dt
                    t0 = time.time()
                    
                    print(f"step {iter_num} | loss: {avg_loss:.4f} | ppl: {perplexity:.2f} | lr: {lr:.6e} | {tokens_per_sec:.0f} tokens/s")
                    
                    if CONFIG['use_wandb']:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/perplexity': perplexity,
                            'train/lr': lr,
                            'train/tokens_per_sec': tokens_per_sec
                        })
                        
                    running_loss = 0.0
                    
        # Validation
        val_loss = 0.0
        print("Evaluating on validation set...")
        val_loss = evaluate(model, val_loader, device, vocab_size=CONFIG['vocab_size'])
        val_ppl = math.exp(val_loss) if val_loss < 20 else -1
        print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
        
        if CONFIG['use_wandb']:
            wandb.log({
                'val/loss': val_loss,
                'val/perplexity': val_ppl
            })
            
        is_new_best = early_stopper(val_loss)
        
        if is_new_best:
            print("New best model found! Saving checkpoint...")
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, f"{CONFIG['save_dir']}/best_model_finetuned.pt")
            
        if early_stopper.early_stop:
            print("Early stopping triggered. Ending training.")
            print("Restoring best model weights.")
            model.load_state_dict(best_model_weights)
            break
                    
        save_path = f"{CONFIG['save_dir']}/chat_model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Model checkpoint saved at {save_path}")
        
        generate_test(model, CONFIG['tokenizer_path'], device)
        
def generate_test(model, tokenizer_path, device):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model.eval()
    
    # 1. Create Prompt
    prompt = "### User:\nWho is Newton?\n\n### Assistant:\n"
    ids = tokenizer.encode(prompt).ids
    
    # 2. Create Tensor (Crucial Fix: Ensure 2D shape [1, Seq_Len])
    # The extra [] around ids creates the batch dimension
    x = torch.tensor([ids], dtype=torch.long).to(device) 
    
    print("\n--- Test Response ---")
    # Generate 30 tokens
    for _ in range(30):
        with torch.no_grad():
            logits, _ = model(x)
            
            # Pick the last token
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Append (Keeping it 2D)
            x = torch.cat((x, torch.tensor([[next_token]], device=device)), dim=1)
            
            # Stop if EOS
            if next_token == tokenizer.token_to_id("[SEP]"): 
                break
            
    decoded = tokenizer.decode(x[0].tolist())
    print(decoded)
    print("---------------------\n")
    model.train()
