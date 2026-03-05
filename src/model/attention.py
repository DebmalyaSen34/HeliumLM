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
        B, T, C = x.size()
        
        # 1. Project and View
        # ⚠️ FIX: Keep shape as [Batch, Time, Heads, Dim] for RoPE
        # DO NOT TRANSPOSE YET
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape Q, K, V for multi-head attention
        # q: [B, T, C] -> [B, T, n_head, d_head]
        # k: [B, T, C'] -> [B, T, n_kv_head, d_head]
        # v: [B, T, C'] -> [B, T, n_kv_head, d_head]
        q = q.view(B, T, self.n_head, self.d_head)
        k = k.view(B, T, self.n_kv_head, self.d_head)
        v = v.view(B, T, self.n_kv_head, self.d_head)
        
        # 2. Apply RoPE (Now the dimensions align correctly)
        q, k = apply_rotary_pos_emb(q, k, freqs_cis=freqs_cis)
        
        # 3. NOW Transpose for Attention: [B, Heads, T, Dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2) # v didn't need RoPE, but needs transpose
        
        # 4. GQA Repeat
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_mask = self.mask[:, :, :T, :T].bool()
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(self.d_head))
            mask_slice = self.mask[:, :, :T, :T]
            att = att.masked_fill(mask_slice == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.output_proj(y))