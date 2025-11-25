import torch
import os
import time
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
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu' 
}

def load_model(model_path: str) -> TinySLM:
    """Loads the trained model from disk.

    Args:
        model_path (str): The path to the model.

    Returns:
        TinySLM: A custom SLM created by me
    """
    
    print(f"Loading model from {model_path}...")
    device = CONFIG['device']
    
    # Load the model
    model = TinySLM(CONFIG).to(device)
    
    # Load the model weights
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict=state_dict)
    model.eval()
    
    print("Model loaded successfully.")
    return model

def generate(model: TinySLM, tokenizer: Tokenizer, prompt: str, max_new_token = 50, temperature=0.7):
    """Generates text using top-k sampling.

    Args:
        model (TinySLM): The trained model.
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding.
        prompt (str): The input text prompt to start generation.
        max_new_token (int, optional): The maximum number of new tokens to generate. Defaults to 50.
        temperature (float, optional): The temperature for sampling. Defaults to 0.7.
    """
    
    device = CONFIG['device']
    
    # Encode the prompt
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long).to(device)
    
    # Generation loop
    start_time = time.time()
    
    for _ in range(max_new_token):
        
        # Forward pass
        # In reality, we would cache KV states for faster generation
        # But our SWA makes it fast anyways
        with torch.no_grad():
            logits, _ = model(x)
            
        # Focus on the last token's logits
        logits = logits[0, -1, :] / temperature
        
        # Top-k sampling
        # Get the top k logits and their indices
        v, _ = torch.topk(logits, k=10)
        logits[logits < v[-1]] = -float('inf')
        
        # Convert to probabilities and sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Append to sequence
        x = torch.cat((x, torch.tensor([[next_token]], device=device)), dim=1)
        
        # Stop if EOS token is generated
        if next_token == tokenizer.token_to_id("[SEP]"):
            break
        
    end_time = time.time()
    
    # Decode
    output_text = tokenizer.decode(x[0].tolist())
    tokens_gen = x.shape[1] - len(ids)
    tps = tokens_gen / (end_time-start_time)
    
    return output_text, tps

if __name__ == "__main__":
    
    tokenizer_path = "data/tokenizer/tiny_slm_tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    model = load_model(model_path="src/model/best_model.pt")
    
    while True:
        prompt = input("\nUser: ")
        if prompt.lower() == "exit":
            break
        
        response, _ = generate(model, tokenizer, prompt)
        print(f"TinySLM: {response}")