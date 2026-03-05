import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precomputes the frequency tensor for complex exponentials (cis).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # Output is complex64: cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes freqs_cis (T, d_head/2) to match x (B, T, n_head, d_head/2)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_pos_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Applies RoPE to Query and Key matrices.
    xq, xk: [B, T, n_head, d_head]
    freqs_cis: [T, d_head/2] (Complex)
    """
    # Reshape for broadcast: [B, T, n_head, d_head/2, 2] -> view as complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs to match [1, T, 1, d_head/2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    # Rotate (Complex Multiplication)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

# ==========================================
# 2. Layers (Norm, MLP)
# ==========================================

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

# class SquaredReLU(nn.Module):
#     """Squared ReLU Activation Function."""
#     def forward(self, x):
#         return torch.pow(F.relu(x), 2)
    
class DepthWiseConv1d(nn.Module):
    """Depth-wise 1D convulation for local context in attention"""
    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv=nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=d_model
        )
    
    def forward(self, x):
        # x: [B, T, d_model]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

# ==========================================
# 3. Attention Mechanism
# ==========================================

class EfficientAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv_head: int, window_size: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.d_head = d_model // n_head
        self.max_seq_len = max_seq_len
        self.use_primer=True
        
        self.window_size = window_size if window_size is not None else max_seq_len

        self.n_rep = self.n_head // self.n_kv_head
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=False)

        if self.use_primer:
            self.dw_conv_q = DepthWiseConv1d(d_model, kernel_size=3)
            self.dw_conv_k = DepthWiseConv1d(self.n_kv_head * self.d_head, kernel_size=3)
            self.dw_conv_v = DepthWiseConv1d(self.n_kv_head * self.d_head, kernel_size=3)
        
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        
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

        if self.use_primer:
            q = self.dw_conv_q(q).contiguous()
            k = self.dw_conv_k(k).contiguous()
            v = self.dw_conv_v(v).contiguous()

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

# ==========================================
# 4. Decoder Block
# ==========================================

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
        
        # self.mlp = SwiGLUMLP(d_model=d_model, expansion_factor=mlp_ratio, dropout=dropout)
        self.mlp = MLP(config)
        
        self.input_layernorm = RMSNorm(d_model)
        self.post_attn_layernorm = RMSNorm(d_model)
        
    def forward(self, x, freqs_cis):
        # Pre-Norm
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis)
        x = x + self.mlp(self.post_attn_layernorm(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        mlp_ratio = config.get('mlp_ratio', 4.0)
        d_ff = int(d_model * mlp_ratio)
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# ==========================================
# 5. The Main Model (TinySLM)
# ==========================================

class TinySLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.max_seq_len = config['max_seq_len']
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.dropout = nn.Dropout(config.get('dropout', 0.0))
        
        self.layers = nn.ModuleList()
        for _ in range(config['n_layers']):
            self.layers.append(DecoderBlock(config))
            
        self.norm = RMSNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Weight Tying
        self.token_embedding.weight = self.output.weight
        
        # Initialize RoPE Cache
        # We precompute for max_seq_len * 2 just to be safe during inference/extrapolation
        freqs_cis = precompute_freq_cis(
            dim=self.d_model // config['n_head'],
            end=self.max_seq_len * 2,
            theta=10000.0
        )
        self.register_buffer("freqs_cis", freqs_cis)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Input Embedding
        x = self.token_embedding(idx)
        x = self.dropout(x)
        
        # Get RoPE frequencies for current sequence length
        freqs_cis = self.freqs_cis[:T]
        
        # Run Layers
        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis)
            
        # Final Norm
        x = self.norm(x)
        
        # Output Logits
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            # Flatten for Cross Entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=None):
        """
        Generation loop for inference.
        """
        for _ in range(max_new_tokens):
            # Crop context if it becomes too long
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
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx