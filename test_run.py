import torch
import time

from src.model.slm import TinySLM

def count_parameters(model: torch.nn.Module) -> str:
    """Helper to count learnable parameters.
    Returns total parameters in formatted string (eg., "15.2M")

    Args:
        model (torch.nn.Module): The model to count parameters for.
    """
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        formatted = f"{total_params / 1e9:.2f}B"
    elif total_params >= 1e6:
        formatted = f"{total_params / 1e6:.2f}M"
    else:
        formatted = f"{total_params / 1e3:.2f}K"
    
    return total_params, formatted

def run_diagnostics():
    
    # 1. The Configuration
    
    config = {
        'vocab_size': 32000,   # Small vocab (Llama uses 32k, GPT-4 uses 100k+)
        'd_model': 256,        # Hidden size. TinyBERT is 312. We use 256 for powers of 2 efficiency.
        'n_layers': 8,         # Depth. Deep enough to learn reasoning.
        'n_head': 8,           # 8 Query heads.
        'n_kv_head': 2,        # GQA: 4 Query heads share 1 KV head. (75% memory reduction here!)
        'window_size': 64,     # SWA: Only look at last 64 tokens.
        'max_seq_len': 512,    # Maximum text length the model can handle.
        'mlp_ratio': 2.5       # SwiGLU expansion. 2.5 * 256 = 640 hidden size.
    }
    
    print(f"--- Initializing TinySLM Configuration ---")
    print(f"Dimensions: {config['d_model']}")
    print(f"Layers: {config['n_layers']}")
    print(f"GQA Ratio: {config['n_head']} Queries : {config['n_kv_head']} KV Heads")
    
    # 2. Model Initialization
    
    model = TinySLM(config=config)
    
    total, fmt = count_parameters(model)
    print(f"\n--- Model Statistics ---")
    print(f"Total Parameters: {fmt}")
    
    # Calculate Theoretical Memory Usage (FP32 = 4 bytes per param)
    size_mb = (total * 4) / (1024 ** 2)
    print(f"Theoretical Memory Usage (FP32): {size_mb:.2f} MB")
    
    # 3. The Forward Pass Test
    
    print(f"\n--- Running Forward Pass Test ---")
    
    # Create a dummy data batch of 4 sentences, each 128 tokens long
    batch_size = 4
    seq_len = 128
    
    # Random integers between 0 and vocab_size
    dummy_input = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    dummy_targest = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Time the inference
    start_time = time.time()
    
    # Forward pass
    logits, loss = model(dummy_input, dummy_targest)
    
    end_time = time.time()
    
    print(f"Output Logits Shape: {logits.shape}") # Should be [4, 128, 32000]
    print(f"Loss Value: {loss.item():.4f}")
    print(f"Inference Time: {(end_time - start_time)*1000:.2f} ms")
    
    # 4. Verification of Efficiency
    
    # Check if weights tying worked
    # The embedding weights and output projection weights should be the same object
    is_tied = model.token_embedding.weight is model.output.weight
    print(f"\nWeight Tying Active? {'✅ YES' if is_tied else '❌ NO'}")
    
    if is_tied:
        print("(This saved you approx. 8.2 Million parameters!)")
        
if __name__ == "__main__":
    run_diagnostics()