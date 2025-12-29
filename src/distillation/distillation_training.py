import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import time
import os
import copy
import wandb
import json
import threading
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# Local Imports
from src.model.slm import TinySLM
from src.data.dataset import TextBookDataset
from src.data.tokenizer import train_tokenizer
from src.distillation.distillationLoss import DistillationLoss
from ...utils import get_lr, evaluate, EarlyStopping

# Hyperparameters and Configurations
try:
    with open('config/heliumLM-nano-gpt2-config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Configuration file 'config/heliumLM-nano-gpt2-config.json' not found.")

# Teacher Model Config
TEACHER_NAME = "gpt2"
DISTILL_TEMP = 2.0
DISTILL_ALPHA = 0.5

#* Training Loop
def train():
    
    # Setup
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    print(f"Using device: {device}")
    mode = CONFIG['train_mode']
    print(f"Training mode: {mode}")

    CONFIG['vocab_size'] = 50257  # GPT-2 vocab size

    #* Limit max_seq_len to GPT-2's max length
    original_seq_len = CONFIG.get('max_seq_len', 512)
    CONFIG['max_seq_len'] = min(original_seq_len, 1024)  # GPT-2 max length
    print(f"⚠️  Limiting max_seq_len to {CONFIG['max_seq_len']} (was {original_seq_len})")

    #* Ensure window_size matches max_seq_len to avoid mask mismatch errors
    if 'window_size' in CONFIG and CONFIG['window_size'] < CONFIG['max_seq_len']:
        print(f"⚠️  Increasing window_size from {CONFIG['window_size']} to {CONFIG['max_seq_len']} to match sequence length")
        CONFIG['window_size'] = CONFIG['max_seq_len']

    # Initialize Hugging Face API
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # 1. Load Teacher model & tokenizer

    print("Loading Teacher Model (GPT-2)...")
    teacher_model = GPT2LMHeadModel.from_pretrained(TEACHER_NAME).to(device)
    teacher_model.eval()

    # Freeze Teacher weights
    for param in teacher_model.parameters():
        param.requires_grad = False

    print("Loading Teacher Tokenizer...")
    teacher_tokenizer = GPT2Tokenizer.from_pretrained(TEACHER_NAME)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    print("Saving Teacher Tokenizer...")
    os.makedirs(os.path.dirname(CONFIG['tokenizer_path']), exist_ok=True)
    teacher_tokenizer.save_pretrained(os.path.dirname(CONFIG['tokenizer_path']))
    print(f"Teacher Tokenizer saved to {CONFIG['tokenizer_path']}")

    # 2. Setup Dataset

    if mode == "huggingface":
        train_source = CONFIG['hf_dataset_name']
        print(f"Training mode: Hugging Face Dataset - {train_source}")
        if teacher_tokenizer is None:
            raise FileNotFoundError("Tokenizer not found. Please provide a tokenizer for Hugging Face datasets.")
    else:
        train_source = CONFIG['train_file']
        print(f"Training mode: Local File - {train_source}")
        if not os.path.exists(CONFIG['tokenizer_path']):
            print("Tokenizer not found. Training a new tokenizer...")
            train_tokenizer(
                dataset_name=CONFIG['train_file'],
                voacb_size=CONFIG['vocab_size'],
            )
            print("Tokenizer training complete.")
            
    dataset = TextBookDataset(
        source=train_source,
        tokenizer_path=CONFIG['tokenizer_path'],
        max_length=CONFIG['max_seq_len'],
        tokenizer=teacher_tokenizer,
        cache_dir="/root/data/cache"
    )
            
    train_loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=True if device == "cuda" else False,
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

    # 3. Initialize Student Model
    #     
    model = TinySLM(config=CONFIG).to(device)

    # if torch.__version__[0] == '2':
    #     print('Compiling the model with torch.compile() for optimization...')
    #     # Use max-autotune for L40S (Triton/Inductor backend)
    #     model = torch.compile(model, mode="reduce-overhead")

    optimzer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.95) # Standard for LLMs
    )
    kde_criterion = DistillationLoss(temperature=4.0, alpha=0.5)
    
    scaler = torch.amp.GradScaler(device='cuda' if device=='cuda' else None)
    early_stopper = EarlyStopping(patience=CONFIG['patience'])
    
    # Logging
    if CONFIG['use_wandb']:
        wandb.init(project="tiny_slm_training", config=CONFIG, name="trial-00")
        wandb.watch(model)
        
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Loop variables
    # We must estimate the total steps roughly since it's an IterableDataset
    # Let's assume 1000 samples / 4 batch_size = 250 steps per epoch
    est_steps_per_epoch = len(train_loader)
    max_iters = CONFIG['max_epochs'] * est_steps_per_epoch
    warmup_iters = int(max_iters * 0.1) # 10% warmup
    print(f"Total training steps: {max_iters}, Warmup steps: {warmup_iters}")
    
    iter_num=0
    running_loss=0.0
    
    # We make a copy of the best model weights in RAM
    best_model_weights = copy.deepcopy(model.state_dict())
    
    model.train()
    
    # Start Epochs
    
    # 4. Training Loop

    for epoch in range(CONFIG['max_epochs']):
        print(f"Starting epoch {epoch+1}/{CONFIG['max_epochs']}...")
        t0 = time.time()
        
        for batch_idx, (X, Y) in enumerate(train_loader):

            # Calculate iteration number
            iter_num = epoch * len(train_loader) + batch_idx

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

            # Teacher Forward Pass (No Grad)
            with torch.no_grad():
                # GPT2 forward
                teacher_outputs = teacher_model(input_ids)
                teacher_logits = teacher_outputs.logits

                # Safety: Ensure shapes match (GPT2 might have diff seq len behavior)
                if teacher_logits.shape[1] != input_ids.shape[1]:
                    # Crop or pad teacher logits to match student input length
                    teacher_logits = teacher_logits[:, :input_ids.shape[1], :]
            
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    student_logits, _ = model(input_ids)

                    # Calculate Distillation Loss
                    # Reshape: (Batch * Seq Len, Vocab Size)
                    loss = kde_criterion(
                        student_logits.reshape(-1, CONFIG['vocab_size']),
                        teacher_logits.reshape(-1, CONFIG['vocab_size']),
                        targets.reshape(-1)
                    )
            else:
                student_logits, _ = model(input_ids)
                loss = kde_criterion(
                    student_logits.reshape(-1, CONFIG['vocab_size']),
                    teacher_logits.reshape(-1, CONFIG['vocab_size']),
                    targets.reshape(-1)
                )

            loss = loss / CONFIG['accum_steps']
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            running_loss += loss.item() * CONFIG['accum_steps'] # Scale back for logging
            
            # Optimizer Step (after accum_steps)
            if (batch_idx + 1) % CONFIG['accum_steps'] == 0:
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
                
                # Upload the model checkpoint to Hugging Face every 5000 steps
                if iter_num % 2000 == 0:
                    print("Uploading model checkpoint to Hugging Face hub...")
                    
                    # Save the checkpoints locally first
                    checkpoint_path = f"{CONFIG['save_dir']}/model_iter_{iter_num}.pt"
                    torch.save(model.state_dict(), checkpoint_path)

                    # Use a separate thread to upload so that training is not blocked
                    def _upload_worker(local_path, remote_path, iteration):
                        try:
                            api.upload_file(
                                path_or_fileobj=local_path,
                                path_in_repo=remote_path,
                                repo_id="batmanLovesAI/HeliumLM",
                                repo_type="model"
                            )
                            print(f"Successfully uploaded checkpoint for iteration {iteration}.")
                        except Exception as e:
                            print(f"Failed to upload checkpoint for iteration {iteration}: {e}")
                    
                    # Start upload in a new thread
                    thread = threading.Thread(
                        target=_upload_worker,
                        args=(checkpoint_path, f"checkpoints/model_iter_{iter_num}.pt", iter_num)
                    )
                    thread.start()

                # Logging
                if iter_num % CONFIG['log_interval'] == 0:
                    
                    # Compute Perplexity
                    avg_loss = running_loss / (CONFIG['log_interval']*CONFIG['accum_steps'])
                    # Perplexity is less meaningful for distillation but we log it anyway
                    perplexity = math.exp(avg_loss) if avg_loss < 20 else -1
                    
                    print(f"Step {iter_num} | Distill Loss: {avg_loss:.4f} | LR: {lr:.2e}")

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
                            "train/distill_loss": avg_loss,
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
            
            if CONFIG['use_wandb']:
                wandb.log({
                    "val/loss": val_loss,
                    "val/perplexity": val_ppl
                })
                

            
            # Early Stopping Check
            is_new_best = early_stopper(val_loss)
            
            if is_new_best:
                print("Found New Best Model! Saving checkpoint...")
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, f"{CONFIG['save_dir']}/best_model.pt")
                
                if CONFIG['use_wandb']:
                    artifact = wandb.Artifact(
                        'best-model',
                        type='model'
                    )
                    
                    artifact.add_file(f"{CONFIG['save_dir']}/best_model.pt")
                    
                    wandb.log_artifact(artifact)
                
            if early_stopper.early_stop:
                print("Early stopping triggered. Ending training.")
                print("Restoring best model weights...")
                model.load_state_dict(best_model_weights)
                break
                    
        print(f"Saving Checkpoint for Epoch {epoch+1}...")
        torch.save(model.state_dict(), f"{CONFIG['save_dir']}/model_epoch_{epoch+1}.pt")
        
    print("Training Complete!")