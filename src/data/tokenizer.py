import os
import json
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

def train_tokenizer(dataset_name: str, voacb_size=32000, min_frequency: int =2) -> Tokenizer:
    print("--------- Training Tokenizer ---------")
    
    
    data_dir = "data/raw"
    temp_text_file = os.path.join(data_dir, "training_corpus.txt")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Download '{dataset_name}' and extract text to {temp_text_file}...")
    
    dataset = load_dataset(dataset_name, split="train")
    
    with open(temp_text_file, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, desc="Extracting text"):
            text = item.get('text', item.get('response', ''))
            if text and len(text.strip())>10:
                f.write(text + "\n")
    
    print("Text extraction completed.")
    
    
    # 1. Initialize an empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 2. Setup Trainer
    # Specialized tokens for controlling the model behavior
    special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
    ]
    trainer = BpeTrainer(
        vocab_size=voacb_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True
    )
    
    print("Training BPE tokenizer from local file...")
    tokenizer.train(files=[temp_text_file], trainer=trainer)
    
    # 5. Save the tokenizer
    output_path = "data/tokenizer/tiny_slm_tokenizer.json"
    os.makedirs("data/tokenizer", exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer trained and saved at {output_path}")
    
    os.remove(temp_text_file)
    print("Temporary files cleaned up.")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer(dataset_name="batmanLovesAI/HeliumSLM-Dataset")