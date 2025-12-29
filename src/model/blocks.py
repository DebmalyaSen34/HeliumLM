import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import EfficientAttention

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLUMLP(nn.Module):
    """Swiss Gated Linear Unit (SwiGLU) Feed-Forward Network."""
    def __init__(self, d_model, expansion_factor=2.5, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        fused = gate * up
        return self.dropout(self.down_proj(fused))
    
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        d_model = config['d_model']
        n_head = config['n_head']
        n_kv_head = config.get('n_kv_head', n_head)
        window_size = config.get('window_size', config['max_seq_len'])
        max_seq_len = config['max_seq_len'] # CRITICAL: Extract this
        mlp_ratio = config.get('mlp_ratio', 2.5)
        dropout = config.get('dropout', 0.0)
        
        self.self_attn = EfficientAttention(
            d_model=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            window_size=window_size,
            max_seq_len=max_seq_len, # Pass it here!
            dropout=dropout
        )
        
        self.mlp = SwiGLUMLP(d_model=d_model, expansion_factor=mlp_ratio, dropout=dropout)
        
        self.input_layernorm = RMSNorm(d_model)
        self.post_attn_layernorm = RMSNorm(d_model)
        
    def forward(self, x, freqs_cis):
        # Pre-Norm
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attn_layernorm(x))
        return x