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
    
    # Create cosine and sine components by duplicating frequencies
    #todo: need more explanation here
    emb = torch.cat((freqs, freqs), dim=-1) # Shape: [end, dim]
    return emb.cos(), emb.sin()  # Return cosine and sine components

def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2 :]
    return torch.cat((-x2, x1), dim=-1)

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

def apply_rotary_pos_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """The Actual Rotation. We treat q/k vectors as complex numbers and multiply by the frequencies.

    Args:
        xq (torch.Tensor): x query tensor.
        xk (torch.Tensor): x key tensor.
        cos (torch.Tensor): Cosine components for rotation.
        sin (torch.Tensor): Sine components for rotation.
    """
    

    # Reshape cos and sin for broadcasting
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1])

    # Apply rotation matrix logic
    # x_new = (x*cos) + (rotate_half(x)*sin)
    xq_out = (xq*cos) + (rotate_half(xq)*sin)
    xk_out = (xk*cos) + (rotate_half(xk)*sin)

    return xq_out.type_as(xq), xk_out.type_as(xk)