import torch
from torch.utils.data import IterableDataset
import json
from tokenizers import Tokenizer
from datasets import load_dataset
from typing import List, Tuple

class TextBookDataset(IterableDataset):
    def __init__(self, source: str, tokenizer_path: str, max_length=512):
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        # self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        
        if self.sep_token_id is None:
            raise ValueError("Tokenizer must have a [SEP] token defined.")
        
        # Data loading and tokenization
        print("Initializing PackedDataset...")
        all_tokens = self._load_and_tokenize(source=source)
        
        # Chunking
        print(f"Packing {len(all_tokens):,} tokens into chunks of max length {self.max_length}...")
        
        self.chunks = []
        
        for i in range(0, len(all_tokens)-self.max_length+1, self.max_length):
            self.chunks.append(torch.tensor(all_tokens[i:i+self.max_length], dtype=torch.long))
            
        print(f"Total chunks created: {len(self.chunks)}")
        
    def _load_and_tokenize(self, source: str) -> List[int]:
        """
        Loads data from the source and returns a list of token IDs.

        Args:
            source (str): Source of the dataset. Can be a local file path (jsonl) or a HuggingFace dataset name.

        Returns:
            List[int]: List of token IDs from the tokenizer.
        """
        
        all_ids = []
        
        if source.endswith('.jsonl'):
            print(f"Loading dataset from local file: {source}...")
            
            with open(source, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        text = json.loads(line).get('text', '')
                        if text:
                            all_ids.extend(self.tokenizer.encode(text).ids)
                            all_ids.append(self.sep_token_id)
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"Downloading dataset from HugginFace: {source}...")
            dataset = load_dataset(source, split='train')
            print(f"Dataset downloaded with {len(dataset)} records.")
            
            for item in dataset:
                text = item.get('text', item.get('response', ''))
                if text:
                    all_ids.extend(self.tokenizer.encode(text).ids)
                    all_ids.append(self.sep_token_id)
        return all_ids
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single chunk. For language modeling, the input and target are the same.
        """
        
        chunk = self.chunks[idx]
        return chunk, chunk