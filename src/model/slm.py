import torch
import torch.nn as nn
from torch.nn import functional as F
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
        
        # 4. The Output Head
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # 5. Weight Tying
        # The matrix that turns Tokens -> Vectors is often the transpose of Vectors -> Tokens
        # Sharing them saves memory and improves performance ~20-30%
        self.token_embedding.weight = self.output.weight
        
        # 6. Precompute RoPE frequencies
        # Compute enough for the max context window (eg. 2048)
        self.cos_cached, self.sin_cached = precompute_freq_cis(
            dim=self.d_model//config['n_head'],
            end=config['max_seq_len']
        )
        
    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        
        # 1. Embed Tokens
        x = self.token_embedding(idx) # Shape: [B, T, d_model]
        
        # Apply Dropout
        x = self.dropout(x)
        
        # 2. Prepare RoPE frequencies for current sequence length
        cos = self.cos_cached[:T].to(x.device)
        sin = self.sin_cached[:T].to(x.device)
        
        # 3. Run through Layers
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)
            
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
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature=0.7, top_k:int=None, repitition_penalty=1.0):
        """
        Generates new tokens.

        Args:
            idx (torch.Tensor): LongTensor of shape (batch, seq_len) - The context
            max_new_tokens (int): How many new tokens should be generated
            temperature (float, optional): How creative the model is. Lesser the value more conservative the model is. Defaults to 0.7.
            top_k (int, optional): Strictly limit to top K chances. Defaults to None.
            repitition_penalty (float, optional): Reduces loops. Defaults to 1.0.
        """

        for _ in range(max_new_tokens):
            # Ensure idx is within the model's max sequence length
            idx_cond = idx if idx.size(1) <= self.config['max_seq_len'] else idx[:, -self.config['max_seq_len']:]

            # Forward pass to get logits
            logits, _ = self.forward(idx_cond)

            # Focus only on the last tine step
            logits = logits[:, -1, :] # Becomes (Batch, vocab_size)

            # Apply repetition penalty
            if repitition_penalty != 1.0:
                # Create a mask of all tokens in current context
                for b in range(idx.shape[0]):
                    prev_tokens = set(idx[b].tolist())
                    for token_id in prev_tokens:
                        # If logit is -ve, multiple makes it more -ve
                        # If logit is +ve, dividing makes it smaller
                        if logits[b, token_id] < 0:
                            logits[b, token_id] *= repitition_penalty
                        else:
                            logits[b, token_id] /= repitition_penalty

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                # Find the value of the k-th largest logit
                v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask anything smaller than that value with -infinity
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1) # (Batch, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (Batch, seq_len+1)

        return idx
