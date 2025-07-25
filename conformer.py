
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ffn import FFNModule
from attention import RoPEAttention
from conv_blocks import ConvolutionModule

class ConformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        n_head: int,
        depthwise_conv_kernel_size: int,
        conv_drop: float = 0.1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.1,
        ffn_dropout: float = 0.1,
        bias: bool = False,
        layer_norm: nn.Module = nn.LayerNorm,
        act_fn: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.ffn1 = FFNModule(input_dim, ffn_dim, dropout=ffn_dropout)

        self.self_attn_layer_norm = layer_norm(input_dim)
        self.self_attn = RoPEAttention(
            input_dim,
            num_heads=n_head,
            qkv_bias=bias,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=layer_norm,
        )

        self.conv_module = ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=conv_drop,
            bias=True,
            layer_norm=layer_norm,
            act_fn=act_fn,
        )

        self.ffn2 = FFNModule(input_dim, ffn_dim, dropout=ffn_dropout)
        self.final_layer_norm = layer_norm(input_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)

        x = x + self.self_attn(self.self_attn_layer_norm(x), key_padding_mask=key_padding_mask)

        x = x + self.conv_module(x)

        z = self.final_layer_norm(x + 0.5 * self.ffn2(x))
        return z