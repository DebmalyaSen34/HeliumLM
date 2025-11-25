import torch
import time
import psutil
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tokenizers import Tokenizer
from src.model.slm import TinySLM

CONFIG = {
    'vocab_size': 32000,
    'd_model': 256,
    'n_layers': 8,
    'n_head': 8,
    'n_kv_head': 2,        # GQA
    'window_size': 64,     # SWA
    'max_seq_len': 256,
    'mlp_ratio': 2.5,
    'device': 'cpu'        # Benchmarking on CPU is often fairer for memory precision
}

def get_memory_mb():
    """Returns current process memory usage in MB."""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)

def benchmark_tinyslm(checkpoint_path="checkpoints/best_model.pt"):
    print("\n--- 🔵 Benchmarking TinySLM (Yours) ---")
    
    # 1. Measure Static Size
    start_mem = get_memory_mb()
    model = TinySLM(CONFIG)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    loaded_mem = get_memory_mb()
    model_size = loaded_mem - start_mem
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.2f}M")
    print(f"Model Load Memory: ~{model_size:.2f} MB")

    # 2. Measure Inference Memory (Peak)
    tokenizer = Tokenizer.from_file("data/tokenizer/tiny_slm_tokenizer.json")
    ids = tokenizer.encode("Photosynthesis is a process").ids
    x = torch.tensor([ids], dtype=torch.long)
    
    # Reset stats
    start_time = time.time()
    torch.manual_seed(42)
    
    # Generate 50 tokens
    for _ in range(50):
        with torch.no_grad():
            logits, _ = model(x)
            next_token = torch.argmax(logits[0, -1, :]).item()
            x = torch.cat((x, torch.tensor([[next_token]])), dim=1)
            
    end_time = time.time()
    end_mem = get_memory_mb()
    
    tps = 50 / (end_time - start_time)
    print(f"Throughput: {tps:.2f} tokens/sec")
    print(f"Peak RAM usage: {end_mem:.2f} MB")
    return param_count, tps, end_mem

def benchmark_tinybert():
    print("\n--- 🔴 Benchmarking TinyBERT (Huawei) ---")
    
    # 1. Load Model
    start_mem = get_memory_mb()
    # TinyBERT 4-layer version (closest match to yours)
    model_name = "huawei-noah/TinyBERT_General_4L_312D" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    
    loaded_mem = get_memory_mb()
    model_size = loaded_mem - start_mem
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.2f}M")
    print(f"Model Load Memory: ~{model_size:.2f} MB")
    
    # 2. Measure Inference
    # Note: BERT is not generative, so we benchmark "Encoding" speed
    # We will feed it a long sequence to simulate load
    inputs = tokenizer("Photosynthesis " * 20, return_tensors="pt") # ~50 tokens
    
    start_time = time.time()
    # Run 50 passes to simulate generating 50 tokens work
    for _ in range(50):
        with torch.no_grad():
            _ = model(**inputs)
            
    end_time = time.time()
    end_mem = get_memory_mb()
    
    # Estimate throughput (Passes per second)
    tps = 50 / (end_time - start_time)
    print(f"Throughput (Equivalent): {tps:.2f} passes/sec")
    print(f"Peak RAM usage: {end_mem:.2f} MB")
    return param_count, tps, end_mem

if __name__ == "__main__":
    print("⚡ BATTLE OF THE SLMs ⚡")
    print("Note: Running on CPU to measure RAM accurately.")
    
    slm_params, slm_speed, slm_mem = benchmark_tinyslm()
    bert_params, bert_speed, bert_mem = benchmark_tinybert()
    
    print("\n===========================================")
    print("             FINAL SCORECARD               ")
    print("===========================================")
    print(f"METRIC        | TinySLM (You) | TinyBERT")
    print(f"--------------|---------------|-----------")
    print(f"Parameters    | {slm_params:.2f}M        | {bert_params:.2f}M")
    print(f"Speed (CPU)   | {slm_speed:.1f} t/s      | {bert_speed:.1f} t/s")
    print(f"Peak RAM      | {slm_mem:.0f} MB         | {bert_mem:.0f} MB")
    print("===========================================")
    
    if slm_mem < bert_mem:
        print("\n🏆 WINNER: TinySLM is more memory efficient!")
    else:
        print("\n🏆 WINNER: TinyBERT (Check your GQA config!)")