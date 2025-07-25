import torch
import torch.nn as nn

################
# POS EMBEDDING
################

class RotaryEmbedding(nn.Module):
    """
    Compute rotary positional embeddings (RoPE) for a given head dimension.
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = base

    def forward(self, x, seq_len):
        B, H, N, D = x.shape
        pos  = torch.arange(N, device=x.device)
        inv  = 1.0 / (self.base ** (torch.arange(0, D, 2, device=x.device) / D))
        freqs = pos[:, None] * inv[None, :]
        cosv  = freqs.cos().unsqueeze(1).unsqueeze(1)
        sinv  = freqs.sin().unsqueeze(1).unsqueeze(1)

        xr = x.permute(2, 0, 1, 3).reshape(N, B, H, D // 2, 2)
        e, o = xr[..., 0], xr[..., 1]
        eo = e * cosv - o * sinv
        oo = e * sinv + o * cosv
        xo = torch.stack([eo, oo], dim=-1).reshape(N, B, H, D)
        return xo.permute(1, 2, 0, 3)