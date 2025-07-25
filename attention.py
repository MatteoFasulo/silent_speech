from typing import Optional

import torch
import torch.nn as nn

from timm.layers import Mlp, LayerScale, DropPath

from pos_embedding import RotaryEmbedding

################
# RoPE Attention
################

class RoPEAttention(nn.Module):
    def __init__(self,
            dim: int,
            num_heads=8,
            qkv_bias=False,
            qk_norm: bool = False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer: nn.Module = nn.LayerNorm,
        ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads, self.dim = num_heads, dim
        self.hd    = dim // num_heads
        self.scale = self.hd ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.hd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.hd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope_dim = self.hd
        self.rope_q = RotaryEmbedding(self.rope_dim)
        self.rope_k = RotaryEmbedding(self.rope_dim)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        B, N, D = x.shape # [batch_size, total_number_tokens, embedding_dimension]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE to a subset of the head dimension
        if 0 < self.rope_dim <= self.hd:
            qr, rem_q = q[..., :self.rope_dim], q[..., self.rope_dim:]
            kr, rem_k = k[..., :self.rope_dim], k[..., self.rope_dim:]
            qr = self.rope_q(qr, seq_len=N)
            kr = self.rope_k(kr, seq_len=N)
            q  = torch.cat([qr, rem_q], dim=-1)
            k  = torch.cat([kr, rem_k], dim=-1)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if key_padding_mask is not None:
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(expanded_mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(2, 1).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CustomAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values: Optional[float] = None,
            drop_path: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
            
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x