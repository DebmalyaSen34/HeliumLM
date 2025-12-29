import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .rope import apply_rotary_pos_emb

class EfficientAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, window_size: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head # Number of query heads
        self.n_kv_head = n_kv_head # Number of key/value heads
        self.d_head = d_model // n_head # Dimension per head
        self.max_seq_len = max_seq_len # Maximum sequence length

        #* Sliding window size for local attention
        # If None, full attention is used
        self.window_size = window_size if window_size is not None else max_seq_len
        
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
        
        # Mask Generation
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        if self.window_size < max_seq_len:
            far_history_mask = torch.tril(torch.ones(max_seq_len, max_seq_len), diagonal=-self.window_size)
            final_mask = causal_mask - far_history_mask
        else:
            final_mask = causal_mask

        self.register_buffer("mask", final_mask.view(1, 1, max_seq_len, max_seq_len))
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, sequence length, model dimension (Channels)
        
        # Calculate Q, K, V
        # Q is standard shape: [Batch, Time, 8 heads, 32 dim]
        # ⚠️ FIX: Keep shape as [Batch, Time, Heads, Dim] for RoPE
        # DO NOT TRANSPOSE YET
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head)
        
        # K, V are reduced shape: [Batch, Time, 2 heads, 32 dim]        
        # In standard attention, these would also be 8 heads
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head)
        
        #* Apply RoPE to Query and Key
        q, k = apply_rotary_pos_emb(q, k, freqs_cis=freqs_cis)
        
        # NOW Transpose for Attention: [B, Heads, T, Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # v didn't need RoPE, but transpose anyway
        
        #* Since K and V have fewer heads, we need to repeat them
        # so that they can align with the Q heads during attention computation
        # We take the 2 KV heads and copy them 4 times each to get 8 "virtual" heads
        # This allows the math to work without storing 8 unique heads in memory
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        #* Check if PyTorch has scaled_dot_product_attention (added in 2.4)
        #* If available, it is more optimized than manual softmax + matmul
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_mask = self.mask[:, :, :T, :T].bool()
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0)
        else:
            # Calculate Score
            #* It calculates how much every token relates to every other token
            att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(self.d_head))

            mask = self.mask[:, :, :T, :T]

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