import torch
import torch.nn as nn
from torch.nn import functional as F
from .blocks import DecoderBlock, RMSNorm
from .rope import precompute_freq_cis

# The Main Body
class HeliumLM(nn.Module):
    """A HeliumLM model"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 1. Embeddings
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.max_seq_len = config['max_seq_len']
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(config.get('dropout', 0.0))
        
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
        
        # 4. Weight Tying
        self.token_embedding.weight = self.output.weight

        # 5. Initialize RoPE Cache
        # We precompute the frequencies for max_seq_len * 2 just to be safe during inference/extrapolation
        freqs_cis = precompute_freq_cis(
            dim=self.d_model//config['n_head'],
            end=self.max_seq_len*2,
            theta=10000.0
        )

        self.register_buffer("freqs_cis", freqs_cis)

        # 6. Initialize Weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        
        # 1. Embed Tokens
        x = self.token_embedding(idx) # Shape: [B, T, d_model]
        
        # Apply Dropout
        x = self.dropout(x)
        
        # 2. Get RoPE frequencies for current sequence length
        freqs_cis = self.freqs_cis[:T]
        
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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None, eos_token_id=50256):
        """
        Generates new tokens, stopping if eos_token_id is generated.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds limit
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-K
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # --- THE FIX: Stop if we hit the End of Text token ---
            if idx_next.item() == eos_token_id:
                break
            # -----------------------------------------------------
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx