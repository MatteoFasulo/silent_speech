import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

################
# PATCH MODULES
################
    
class PatchEmbedWaveformKeepChans(nn.Module):
    """Waveform to embedding that maintain the channel dimension"""
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
        pad_to_be_divisible: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pad_to_be_divisible = pad_to_be_divisible
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
        )

    def forward(self, x):
        B, C, T = x.shape

        if self.pad_to_be_divisible:
            # Pad sequence to be divisible by patch_size
            pad_amount = (self.patch_size - (T % self.patch_size)) % self.patch_size
            if pad_amount > 0:
                x = F.pad(x, (0, pad_amount))

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1)) 
        x = rearrange(x, 'B D C t -> B (C t) D')
        return x