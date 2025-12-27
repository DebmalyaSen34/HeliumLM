import torch
import torch.nn as nn
import math

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