import torch.nn as nn
from .blocks import DecoderBlock, RMSNorm
from .rope import precompute_freq_cis

# The Main Body
class TinySLM(nn.Module):
    """A TinySLM model"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 1. Embeddings
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # 2. The Transformer Layers
        # Implementating Block-Wise Weight Sharing
        # If n_unique_layers < n_layers, we reuse the modules
        self.layers = nn.ModuleList()
        n_layers = config['n_layers']
        
        # Create the actual blocks
        for _ in range(n_layers):
            self.layers.append(DecoderBlock(config))
            
        # 3. Final Normalization
        self.norm = RMSNorm(self.d_model)
        
        # 4. The Output Head
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # 5. Weight Tying
        # The matrix that turns Tokens -> Vectors is often the transpose of Vectors -> Tokens
        # Sharing them saves memory and improves performance ~20-30%
        self.token_embedding.weight = self.output.weight
        
        # 6. Precompute RoPE frequencies
        # Compute enough for the max context window (eg. 2048)
        self.freqs_cis = precompute_freq_cis(
            dim=self.d_model // config['n_head'], # Dimension per head
            end=config['max_seq_len'] * 2, # Just to be safe
            theta=10000.0
        )
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Embed Tokens
        x = self.token_embedding(idx) # Shape: [B, T, d_model]
        
        # 2. Prepare RoPE frequencies for current sequence length
        freqs_cis = self.freqs_cis[:T].to(x.device)
        
        # 3. Run through Layers
        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis)
            
        # 4. Final Normalization
        x = self.norm(x)
        
        # 5. Calculate logits (if training)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            # Flatten for cross-entropy [B*T, vocab_size]
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
        return logits, loss