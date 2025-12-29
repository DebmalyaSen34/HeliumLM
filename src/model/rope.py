import torch

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