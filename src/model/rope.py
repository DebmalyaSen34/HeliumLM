import torch

def precompute_freq_cis(dim: int, end: int, theta: float = 10000.0):
    """Precomputes the angles for rotation.
    Drawing a map of all possible positions ahead of time.

    Args:
        dim (int): The dimension of attention head.
        end (int): The maximum sequence length.
        theta (float, optional): The base frequency. Defaults to 10000.0 which is standard for Llama.
    """
    
    # Create a list of frequencies
    # 1 / (theta ^ (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # Create a list of positions: 0, 1, 2, ..., end-1
    t = torch.arange(end, device=freqs.device)
    
    # Compute the outer product to get all position-frequency combinations
    freqs = torch.outer(t, freqs).float() # Shape: [end, dim//2]
    
    # Turn them into polar coordinates (mag 1, angle freqs)
    # Cis = cos + i*sin
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper to match shapes.
    freqs_cis: [max_len, dim // 2]
    x: [Batch, seq_len, head, dim // 2]

    Args:
        freqs_cis (torch.Tensor): Polar coordinates of frequencies.
        x (torch.Tensor): Input tensor.
    """
    
    ndim = x.ndim # Number of dimensions in x
    assert 0 <= 1 < ndim # Ensure x has at least 2 dimensions
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # Check shape compatibility
    
    # Reshape freqs_cis to [1, seq_len, 1, dim // 2] so that it broadcasts over Batch and Head
    shape = [d if i==1 or i==ndim-1 else 1 for i, d in enumerate(x.shape)]
    
    return freqs_cis.view(*shape)

def apply_rotary_pos_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """The Actual Rotation. We treat q/k vectors as complex numbers and multiply by the frequencies.

    Args:
        xq (torch.Tensor): x query tensor.
        xk (torch.Tensor): x key tensor.
        freqs_cis (torch.Tensor): Polar coordinates of frequencies.
    """
    
    # 1. Turn Query and Key into complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Get the specific frequencies for the current sequence length
    freqs_cis = reshape_for_broadcast(freqs_cis=freqs_cis, x=xq_)
    
    # 3. Rotate (Multiply) and convert back to real numbers
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)