import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rotary_pos_emb

class EfficientAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, window_size: int, dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.n_head = n_head # Number of query heads
        self.n_kv_head = n_kv_head # Number of key/value heads
        self.d_head = d_model // n_head # Dimension per head
        self.window_size = window_size # Size of the local attention window
        
        #* The GQA Ratio (Grouped Query Attention Ratio)
        #* If n_head=8 and n_kv_head=2, then n_rep=4
        #* This means 1 K/V head will serve 4 query heads
        self.n_rep = self.n_head // self.n_kv_head
        
        # Q needs full size: (d_model -> d_model)
        #* Why? Because each query head is unique
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        
        #* Instead of mapping (d_model -> d_model), we map (d_model -> d_model / 4)
        # Why? Because each K/V head is shared among multiple query heads
        # This reduces the number of parameters and computation
        self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
        
        #* Output projection to combine heads back to d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
        #* Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Precompute the sliding window + casual mask
        casual_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        window_mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=-window_size+1)
        combined_mask = casual_mask * window_mask
        self.register_buffer("mask", combined_mask)
        
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, model dimension (Channels)
        
        # Calculate Q, K, V
        # Q is standard shape: [Batch, Time, 8 heads, 32 dim]
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2) #* Why transpose? -> [B, n_head, T, d_head] PyTorch's matrix multiplication (@) operates on the last two dimensions, so we need Time and Dim at the end.
        
        # K, V are reduced shape: [Batch, Time, 2 heads, 32 dim]        
        # In standard attention, these would also be 8 heads
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        
        #* Apply RoPE to Query and Key
        q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)
        # Tranpose back to [B, head, T, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        #* Since K and V have fewer heads, we need to repeat them
        # so that they can align with the Q heads during attention computation
        # We take the 2 KV heads and copy them 4 times each to get 8 "virtual" heads
        # This allows the math to work without storing 8 unique heads in memory
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        #* Check if PyTorch has scaled_dot_product_attention (added in 2.4)
        #* If available, it is more optimized than manual softmax + matmul
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_mask = self.mask[:T, :T].to(torch.bool)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            # Calculate Score
            #* It calculates how much every token relates to every other token
            att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(self.d_head))

            mask = self.mask[:T, :T]

            att = att.masked_fill(mask==0, float('-inf'))
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        #* Reshape and Output Projection
        # y: [B, n_head, T, d_head] -> [B, T, n_head, d_head] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        #* Dropout
        y = self.output_proj(y)
        y = self.resid_dropout(y)
        
        return y
    
if __name__ == "__main__":
    model = EfficientAttention(d_model=256, n_head=8, n_kv_head=2, window_size=16)
    x = torch.randn(1, 64, 256) # [Batch, Time, Channels]
    output = model(x)
    
    print(f"Output Shape: {output.shape}") # Should be [1, 64, 256]
    print(f"Param Count: {sum(p.numel() for p in model.parameters())}")