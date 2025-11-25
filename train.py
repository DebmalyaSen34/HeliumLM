import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import time
import os
import copy
import wandb

from src.model.slm import TinySLM
from src.data.dataset import TextBookDataset
from src.data.tokenizer import train_tokenizer

# Hyperparameters and Configurations

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
    'learning_rate': 3e-4, # Peak LR (standard for small models)
    'max_epochs': 100,
    'patience': 3,
    'weight_decay': 0.01,  # AdamW regularization
    'grad_clip': 1.0,      # Prevents exploding gradients
    
    # Data Paths
    'train_file': 'data/raw/hybrid_textbook_data.jsonl',
    'val_file': 'data/raw/validation_textbook.jsonl',
    'tokenizer_path': 'data/tokenizer/tiny_slm_tokenizer.json',
    'save_dir': 'checkpoints',
    
    # Logging
    'use_wandb': False,    # Set to True if you have an account
    'log_interval': 10     # Print every 10 steps
}

# Utils: Cosine Scheduler with Warmup

class EarlyStopping:
    """
    The Watchdog. It counts how many did the validation failed to improve.
    """
    
    def __init__(self, patience: int =3, min_delta: float=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter=0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0 # Reset counter if we improved
            return True # New Best model found
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def get_lr(it: int, max_iters: int, warmup_iters: int, min_lr: float, max_lr: float):
    """
    Calculates the learning rate for the current iteration 'it'.
    Implements Linear Warmup + Cosine Decay.

    Args:
        it (int): Current iteration number
        max_iters (int): Total number of iterations
        warmup_iters (int): Number of warmup iterations
        min_lr (float): Minimum learning rate
        max_lr (float): Maximum learning rate
    """
    
    # Linear Warmup
    if it<warmup_iters:
        return max_lr * (it+1) / warmup_iters
    
    # If we are past the end, return min_lr
    if it>max_iters:
        return min_lr
    
    # Cosine Decay
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1+math.cos(math.pi*decay_ratio))
    
    return min_lr + coeff * (max_lr - min_lr)

def evaluate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device, vocab_size: int) -> float:
    """
    Runs the model on the exam (validation set) without updating the weights

    Args:
        model (torch.nn.Module): The SLM model
        val_loader (torch.utils.data.DataLoader): Validation data loader
        device (torch.device): Device to run the model on (e.g., 'cuda', 'cpu')
        vocab_size (int): Size of the vocabulary
    """
    
    model.eval()
    total_loss = 0
    steps=0
    
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            
            logits, _ = model(X[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                Y[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
            steps+=1
    model.train()
    return total_loss / steps

# Training Loop

def train():
    
    # Setup
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    
    # Ensure Tokenizer Exists
    # If the user hasn't trained a tokenizer yet, do it now
    if not os.path.exists(CONFIG['tokenizer_path']):
        print("Training tokenizer...")
        train_tokenizer(CONFIG['train_file'], CONFIG['vocab_size'])
        
    # Train Data Loader
    train_ds = TextBookDataset(
        file_path=CONFIG['train_file'],
        tokenizer_path=CONFIG['tokenizer_path'],
        max_length=CONFIG['max_seq_len']
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=True if device == "cuda" else False
    )
    
    if os.path.exists(CONFIG['val_file']):
        # Validation Data Loader
        val_ds = TextBookDataset(
            file_path=CONFIG['val_file'],
            tokenizer_path=CONFIG['tokenizer_path'],
            max_length=CONFIG['max_seq_len']
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            pin_memory=True if device == "cuda" else False
        )
    else:
        print("Warning: Validation file not found. Skipping validation.")
        val_loader = None
    
    # Model Initialization
    model = TinySLM(config=CONFIG).to(device)
    optimzer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.95) # Standard for LLMs
    )
    
    # Enable AMP (Automatic Mixed Precision) if using CUDA
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
    
    early_stopper = EarlyStopping(patience=CONFIG['patience'])
    
    # Logging
    if CONFIG['use_wandb']:
        wandb.init(project="tiny_slm_training", config=CONFIG, name="trial-00")
        wandb.watch(model)
        
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Loop variables
    # We must estimate the total steps roughly since it's an IterableDataset
    # Let's assume 1000 samples / 4 batch_size = 250 steps per epoch
    est_steps_per_epoch = 1000 // CONFIG['batch_size']
    max_iters = CONFIG['max_epochs'] * est_steps_per_epoch
    warmup_iters = int(max_iters * 0.1) # 10% warmup
    
    iter_num=0
    running_loss=0.0
    
    # We make a copy of the best model weights in RAM
    best_model_weights = copy.deepcopy(model.state_dict())
    
    model.train()
    
    # Start Epochs
    
    for epoch in range(CONFIG['max_epochs']):
        print(f"Starting epoch {epoch+1}/{CONFIG['max_epochs']}...")
        t0 = time.time()
        
        for batch_idx, (X, Y) in enumerate(train_loader):
            
            # Update the learning rate: Cosine Scheduler
            lr = get_lr(iter_num, max_iters, warmup_iters, 3e-5, CONFIG['learning_rate'])
            for param_group in optimzer.param_groups:
                param_group['lr'] = lr
                
            # Move data to device
            X, Y = X.to(device), Y.to(device)
            
            # Create targets (Next Token Prediction)
            # In input:'the cat sat', target:'cat sat on'
            
            input_ids = X[:, :-1]
            targets = Y[:, 1:]
            
            # Forward Pass (with AMP if CUDA)
            if scaler:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(input_ids)
                    
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, CONFIG['vocab_size']),
                        targets.reshape(-1)
                    )
            else:
                # MPS
                logits, _ = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, CONFIG['vocab_size']),
                    targets.reshape(-1)
                )
                
            # Gradient Accumulation Scaling
            # If we want a virtual bach size of 32 but can only fit 4, we simple divide loss by 8. Summing 8 small gradients = 1 big gradient
            loss = loss / CONFIG['accum_steps']
            
            # Backward Pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            running_loss += loss.item() * CONFIG['accum_steps'] # Scale back for logging
            
            # Optimizer Step (after accum_steps)
            if (batch_idx + 1) % CONFIG['accum_steps'] == 0:
                # Capture gradient norm
                if scaler:
                    scaler.unscale_(optimzer)
                    
                # clip_grad_norm_ returns the norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
                
                # Update Weights
                if scaler:
                    scaler.step(optimzer)
                    scaler.update()
                else:
                    optimzer.step()
                    
                # Flush gradients
                optimzer.zero_grad(set_to_none=True)
                
                iter_num +=1
                
                # Logging
                if iter_num % CONFIG['log_interval'] == 0:
                    
                    # Compute Perplexity
                    avg_loss = running_loss / CONFIG['log_interval']
                    perplexity = math.exp(avg_loss) if avg_loss < 20 else -1
                    
                    # Calculate Tokens per second (throughput)
                    # We processed (batch_size * seq_len * accum_steps * log_interval) tokens
                    tokens_processed = (CONFIG['batch_size'] * CONFIG['max_seq_len'] * CONFIG['accum_steps'] * CONFIG['log_interval'])
                    
                    dt = time.time() - t0
                    tokens_per_sec = tokens_processed / dt
                    t0 = time.time()
                    
                    # Calculate Memory (if CUDA)
                    mem_usage = 0
                    if device == 'cuda':
                        mem_usage = torch.cuda.max_memory_allocated()/1024**2 # in MB
                        torch.cuda.reset_peak_memory_stats() # reset for next logging
                    
                    print(f"step {iter_num} | loss: {avg_loss:.4f} | ppl: {perplexity:.1f} | "
                          f"norm: {grad_norm:.2f} | mem: {mem_usage:.0f}MB | {tokens_per_sec:.0f} tok/s")
                    
                    if CONFIG['use_wandb']:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/perplexity": perplexity,
                            "train/learning_rate": lr,
                            "train/grad_norm": grad_norm,
                            "perf/tokens_per_sec": tokens_per_sec,
                            "perf/memory_MB": mem_usage
                        })
                    running_loss = 0.0
                    
        # Validation
        val_loss = 0.0
        if val_loader is not None:
            print("Running Validation...", end="")
            val_loss = evaluate(model, val_loader, device, vocab_size=CONFIG['vocab_size'])
            val_ppl = math.exp(val_loss) if val_loss <20 else -1
            print(f" Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}")
            
            # Early Stopping Check
            is_new_best = early_stopper(val_loss)
            
            if is_new_best:
                print("Found New Best Model! Saving checkpoint...")
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, f"{CONFIG['save_dir']}/best_model.pt")
                
            if early_stopper.early_stop:
                print("Early stopping triggered. Ending training.")
                print("Restoring best model weights...")
                model.load_state_dict(best_model_weights)
                break
                    
        print(f"Saving Checkpoint for Epoch {epoch+1}...")
        torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model_epoch_{epoch+1}.pt")
        
    print("Training Complete!")
    
if __name__ == "__main__":
    train()
