import torch
import torch.utils.data
from datasets import load_dataset
from tokenizers import Tokenizer

class AlpacaChatDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_path: str, max_seq_len: int=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        
        # Load dataset fom HuggingFace
        print("Downloading Alpaca dataset...")
        hf_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        
        self.samples = []
        for item in hf_dataset:
        
            if item['input']:
                prompt = f"{item['instruction']}\nInput: {item['input']}"
            else:
                prompt = item['instruction']
        
            text = (
                f"### User:\n{prompt}\n\n"
                f"### Assistant:\n{item['output']}"
            )
            
            # Tokenizer
            encoded = self.tokenizer.encode(text)
            ids = encoded.ids
            
            if self.pad_token_id:
                ids.appenmd(self.sep_token_id) 
            self.samples.append(ids)
            
        print(f"Loaded {len(self.samples)} samples from Alpaca dataset.")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]

        ids = self.samples[idx]

        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]

        padding_len = self.max_seq_len - len(ids)
        if padding_len>0:
            ids = ids + [self.pad_token_id]*padding_len
        
        x = torch.tensor(ids, dtype=torch.long)
        
        # Target is same as input for casual language modeling
        # In advance instruction tuning, the user prompt will be masked so that the model only learns to generate the response
        return x, x