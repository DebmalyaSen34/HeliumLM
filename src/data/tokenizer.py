from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def train_tokenizer(input_file="data/raw/hybrid_textbook_data.jsonl", voacb_size=32000):
    print("--------- Training Tokenizer ---------")
    
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
        special_tokens=special_tokens
    )
    
    # 3. Stream data from file so that we dont load RAM
    # Read 'text' field from each json line
    files = [input_file]
    
    # 4. Train
    tokenizer.train(files, trainer)
    
    # 5. Save the tokenizer
    os.makedirs("data/tokenizer", exist_ok=True)
    tokenizer.save("data/tokenizer/tiny_slm_tokenizer.json")
    print("Tokenizer trained and saved at data/tokenizer/tiny_slm_tokenizer.json")
    
    return tokenizer

if __name__ == "__main__":
    train_tokenizer()