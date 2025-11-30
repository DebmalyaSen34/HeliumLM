import torch
import torch.utils.data
from datasets import load_dataset
from tokenizers import Tokenizer
import json

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer_path: str, max_seq_len: int=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        
        self.samples = []
        
        # Load dataset fom HuggingFace
        # Limit to 2000 samples for so that it doesn't drown newton knowledge samples
        print("Downloading dataset...")
        hf_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        for i, item in enumerate(hf_dataset):
            if i>2000:
                break
            self.add_sample(item["instruction"], item["response"], item["context"])
        
        # Load knowledge injection dataset
        print("Loading knowledge injection dataset...")
        with open("data/raw/injection_knowledge_dataset.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                
                # We add these multiple times to increase their presence in the training data
                for _ in range(5):
                    self.add_sample(item["instruction"], item["response"], item.get('context', ''))
        print(f"Total samples in ChatDataset: {len(self.samples)}")
        
    def __len__(self):
        return len(self.samples)
    
    def add_sample(self, instruction: str, response: str, context: str):
        # Format:
        # ### User:
        # [Instruction]
        # Context: [Context] (Optional)
        #
        # ### Assistant:
        # [Response]
        
        ctx_str = f"\nContext: {context}" if context else ""
        text = f"### User:\n{instruction}{ctx_str}\n\n### Assistant:\n{response}"
        
        # Tokenize
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        
        # Add EOS token at the end
        if self.sep_token_id is not None:
            ids.append(self.sep_token_id)
            
        self.samples.append(ids)
    
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