
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ffn import FFNModule
from attention import RoPEMHA
from conv_blocks import ConvolutionModule

class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_head: int,
        depthwise_conv_kernel_size: int,
        conv_drop: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_dropout: float = 0.0,
        bias: bool = True,
        layer_norm: nn.Module = nn.LayerNorm,
        act_fn: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.mha_layer = RoPEMHA(
            num_heads=n_head,
            embed_dim=d_model,
            dropout=attn_drop,
        )

        self.convolution_module = ConvolutionModule(
            input_size=d_model,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=conv_drop,
            bias=bias,
            layer_norm=layer_norm,
            act_fn=act_fn,
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            FFNModule(
                hidden_dim=d_ffn,
                input_dim=d_model,
                dropout=ffn_dropout,
                layer_norm=layer_norm,
                act_fn=act_fn,
            ),
            nn.Dropout(ffn_dropout),
        )
        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            FFNModule(
                hidden_dim=d_ffn,
                input_dim=d_model,
                dropout=ffn_dropout,
                layer_norm=layer_norm,
                act_fn=act_fn,
            ),
            nn.Dropout(ffn_dropout),
        )

        self.norm1 = layer_norm(d_model)
        self.norm2 = layer_norm(d_model)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, 
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)
        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # multi-head attention module
        skip = x
        x = self.norm1(x)

        x, self_attn = self.mha_layer(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        x = x + skip
        # convolution module
        x = x + self.convolution_module(x, conv_mask)
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn