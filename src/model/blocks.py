import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import EfficientAttention

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    Why: It is faster than LayerNorm because it skips calculatin the mean.
    It only calculates variance.
    """
    
    def __init__(self, dim, eps = 1e-6):
        super().__init__()

        self.eps = eps # Prevents division by zero
        
        # The learnable weight parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        
        # x.pow(2) -> squares each element
        # .mean(-1, keepdim=True) -> mean over the last dimension (features)
        # + self.eps -> add epsilon for numerical stability
        # torch.rsqrt(...) -> reciprocal of the square root
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        
        # Convert to float32 for stability during normalization and later convert it back to original dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class SwiGLUMLP(nn.Module):
    """
    Swiss Gated Linear Unit (SwiGLU) Feed-Forward Network.
    Why: It learns efficiently and better than standard ReLU FFNs.
    But it has 3 layers instead of 2.
    """
    
    def __init__(self, d_model, expansion_factor=2.5, dropout: float = 0.0):
        super().__init__()
        
        # Standard Transformer use expansion_factor of 4
        # SLMs use ~2.6 (Llama) or lower to save memory
        hidden_dim = int(d_model * expansion_factor)
        
        # 1. Gate projection - Determines which information to pass through
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # 2. Up projection - Expands the dimensionality
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        # 3. Down projection - Reduces back to d_model
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply the SwiGLU activation: (SiLU(gate) * up_proj) -> down_proj
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Element-wise multiplication (gating mechanism)
        fused = gate * up
        
        # Apply down projection and dropout
        return self.dropout(self.down_proj(fused))
    
class DecoderBlock(nn.Module):
    """A Single Transformer Block.
    
    Flow: Input -> RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual
    """
    
    def __init__(self, config):
        super().__init__()
        
        d_model = config['d_model']
        n_head = config['n_head']
        n_kv_head = config.get('n_kv_head', n_head) # Default to n_head if not specified
        window_size = config.get('window_size', config['max_seq_len']) # Default to full sequence
        max_seq_len = config['max_seq_len']
        mlp_ratio = config.get('mlp_ratio', 2.5) # Default to 2.5 for efficiency
        dropout = config.get('dropout', 0.0)
        
        # 1. Attention Engine
        self.self_attn = EfficientAttention(
            d_model=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            window_size=window_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # 2, The Thinking Engine (FFN)
        self.mlp = SwiGLUMLP(
            d_model=d_model,
            expansion_factor=mlp_ratio,
            dropout=dropout
        )
        
        # 3. Normalization Layers (1 before attention, 1 before FFN)
        self.input_layernorm = RMSNorm(d_model)
        self.post_attn_layernorm = RMSNorm(d_model)
        
    def forward(self, x, cos, sin):
        
        # 1. Attention Block with Residual Connection
        # Norm before attention because it is more stable
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x
        
        # 2. MLP Block with Residual Connection
        residual = x
        x = self.post_attn_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x