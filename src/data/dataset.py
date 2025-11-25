import torch
from torch.utils.data import IterableDataset
import random
import json
from tokenizers import Tokenizer

class TextBookDataset(IterableDataset):
    def __init__(self, file_path: str, tokenizer_path: str, max_length=512):
        self.file_path = file_path
        self.max_length = max_length
        
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        
    def __iter__(self):
        """
        Generator that reads the file, tokenizes dynamically and yields chunks
        """
        buffer_token = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                
                try:
                    record = json.loads(line)
                    text = record['text']
                    
                    # Tokenize
                    encode = self.tokenizer.encode(text)
                    ids = encode.ids
                    
                    # Add [EOS] token using [SEP] as a stand-in
                    ids.append(self.tokenizer.token_to_id("[SEP]"))
                    
                    buffer_token.extend(ids)
                    
                    # When buffer exceeds max_length, yield chunks
                    while len(buffer_token) >= self.max_length:
                        # Slice of a chunk
                        chunk = buffer_token[:self.max_length]
                        buffer_token = buffer_token[self.max_length:]
                        
                        # Prepare Input and Target
                        # Input: [A, B, C, D]
                        # Target: [B, C, D, E] (Next token prediction)
                        
                        # Ideally for training we just return the chunk
                        # The training loop will handle shifting for next token prediction
                        x = torch.tensor(chunk, dtype=torch.long)
                        yield x, x # Target is same as input for next token prediction
                except json.JSONDecodeError:
                    continue