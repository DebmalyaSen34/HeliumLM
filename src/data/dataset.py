import torch
from torch.utils.data import Dataset
import json
import pickle
import hashlib
from pathlib import Path
from tokenizers import Tokenizer
from datasets import load_dataset
from typing import List, Tuple, Union
from transformers import PreTrainedTokenizerBase

class HeliumLMDataset(Dataset):
    def __init__(self, source: str, tokenizer_path: str, max_length=512, is_half=False, complexity_level="all", tokenizer: Union[Tokenizer, PreTrainedTokenizerBase] = None, cache_dir: str = "data/cache"):
        """Dataset class for HeliumLM

        Args:
            source (str): A local dataset file or a Huggingface dataset id.
            tokenizer_path (str): Path to your tokekinzer.
            max_length (int, optional): Maximum length of the sequence to be tokenized once. Defaults to 512.
            is_half (bool, optional): Set True if you want only first half of the dataset. Defaults to False.
            complexity_level (str, optional): "simple" (<500 chars), "complex" (>=500 chars), or "all". Defaults to "all".
            tokenizer (Union[Tokenizer, PreTrainedTokenizerBase], optional): Tokenizer that is being used, can be something trained from scratch or from huggingface. Defaults to None.
            cache_dir (str, optional): Directory to store cached tokenized dataset. Defaults to "data/cache".

        Raises:
            ValueError: If tokenizer or tokenizer path is not found.
        """

        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer.from_file(tokenizer_path)
        self.complexity_level = complexity_level
        # Determine tokenizer type
        # 'convert_tokens_to_ids' is a method of HuggingFace's PreTrainedTokenizerBase
        self.is_hf_tokenizer = hasattr(self.tokenizer, 'convert_tokens_to_ids')

        if self.is_hf_tokenizer:
            # HuggingFace transformers tokenizer
            self.sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            # Get model max length if defined
            model_max_length = getattr(self.tokenizer, 'model_max_length', 1024)
            self.max_length = min(max_length, model_max_length)

            tokenizer_name = self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else 'hf_tokenizer'
            print(f"Using HuggingFace tokenizer with model max length {model_max_length}.")
        else:
            # HuggingFace tokenizers.Tokenizer
            self.sep_token_id = self.tokenizer.token_to_id('[SEP]')
            self.pad_token_id = self.tokenizer.token_to_id('[PAD]')
            self.max_length = max_length
            tokenizer_name = "custom_heliumLM_tokenizer"
        
        if self.sep_token_id is None:
            raise ValueError("Tokenizer must have a [SEP] token defined.")
        
        # Create a cache directory
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Generate unique cache filename based on dataset, tokenizer, and config
        cache_key = f"{source}_{tokenizer_name}_{self.max_length}_{is_half}"
        cache_path = hashlib.md5(cache_key.encode()).hexdigest() # Unique hash
        self.cache_file = Path(cache_dir) / f"tokenized_{cache_path}.pkl"

        # Try to load from cache
        if self.cache_file.exists():
            print(f"Loading tokenized dataset from cache: {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} cached chunks")
        else:
            print("Cache not found. Processing dataset...")

            # Data loading and tokenization
            print("Initializing PackedDataset...")
            all_tokens = self._load_and_tokenize(source=source, is_half=is_half)

            # Chunking
            print(f"Packing {len(all_tokens):,} tokens into chunks of max length {self.max_length}...")
            
            self.chunks = []
            
            for i in range(0, len(all_tokens)-self.max_length+1, self.max_length):
                self.chunks.append(torch.tensor(all_tokens[i:i+self.max_length], dtype=torch.long))
                
            print(f"Total chunks created: {len(self.chunks)}")

            # Save to cache
            print(f"Saving tokenized dataset to cache: {self.cache_file}")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("Cache saved successfully!")

    def _tokenize_text(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=self.max_length)
        else:
            return self.tokenizer.encode(text).ids
        
    def _load_and_tokenize(self, source: str, is_half=False) -> List[int]:
        """
        Loads data from the source and returns a list of token IDs.

        Args:
            source (str): Source of the dataset. Can be a local file path (jsonl) or a HuggingFace dataset name.

        Returns:
            List[int]: List of token IDs from the tokenizer.
        """
        
        all_ids = []

        THRESHOLD = 500  # Character threshold to differentiate simple and complex texts

        def is_valid(text: str) -> bool:
            if not text:
                return False
            if self.complexity_level == "simple" and len(text) >= THRESHOLD:
                return len(text)<THRESHOLD
            elif self.complexity_level == "complex":
                return len(text)>=THRESHOLD
            return True
        
        # Check if source is a local file
        if source.endswith('.jsonl'):
            print(f"Loading dataset from local file: {source}...")
            
            with open(source, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')

                        # Filter based on complexity level
                        if is_valid(text):
                            all_ids.extend(self._tokenize_text(text))
                            all_ids.append(self.sep_token_id)
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"Downloading dataset from HugginFace: {source}...")
            dataset = load_dataset(source, split=f'train{"[:50%]" if is_half else ""}')
            print(f"Dataset downloaded with {len(dataset)} records.")
            
            # Iterate through dataset and tokenize
            for idx, item in enumerate(dataset):
                if idx % 10000 == 0:
                    print(f"Processed {idx}/{len(dataset)} records...")
                text = item.get('text', item.get('response', ''))
                
                # Filter based on complexity level
                if is_valid(text):
                    all_ids.extend(self._tokenize_text(text))
                    all_ids.append(self.sep_token_id)
        print(f"Tokenizeation completed. Total tokens: {len(all_ids):,}")
        return all_ids
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single chunk. For language modeling, the input and target are the same.
        """
        
        chunk = self.chunks[idx]
        return chunk, chunk