import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from architecture_gaddy import ResBlock
from transformer_gaddy import LearnedRelativePositionalEmbedding


class LRPEAttention(nn.Module):
    """
    Multi Head Attention with Learned Relative Positional Encoding (LRPE) applied to the logits.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        relative_positional_distance: int = 100,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads, self.dim = num_heads, dim
        self.hd = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.relative_positional = LearnedRelativePositionalEmbedding(
            relative_positional_distance, num_heads, self.hd, True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch_size, length, d_model]
        Returns:
          A single tensor containing the output from this layer
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Attention is batch-first here: q has shape [B, H, L, Dh].
        scale_factor = 1 / math.sqrt(q.size(-1))
        logits = q @ k.transpose(-2, -1) * scale_factor

        # The shared Gaddy LRPE implementation expects [L, B * H, Dh].
        q_pos = q.permute(0, 2, 1, 3)  # [B, N, n_h, h_d]
        b, l, h, d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l, b * h, d))
        # LRPE returns [B * H, L, L]; restore [B, H, L, L].
        assert position_logits.shape == (b * h, l, l)
        position_logits = position_logits.view(b, h, l, l)
        logits = logits + position_logits

        probs = F.softmax(logits, dim=-1)
        probs = self.attn_drop(probs)

        out = (probs @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.1,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class CustomAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        attention_type: str = "lrpe",
    ) -> None:
        super().__init__()
        if attention_type in ("lrpe", "rope"):
            # "rope" is retained as a legacy config value, but now resolves
            # to LRPE. RoPE is no longer implemented in this model.
            self.attn = LRPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                relative_positional_distance=100,
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type!r}")
        self.norm1 = norm_layer(dim)
        ffn_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=ffn_dim,
            out_features=dim,
            dropout=proj_drop,
            act_layer=act_layer,
        )
        self.dropout1 = nn.Dropout(proj_drop)
        self.dropout2 = nn.Dropout(proj_drop)
        self.norm2 = norm_layer(dim)

        self.activation = act_layer()

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.mlp(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class EMGTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outs: int,
        num_aux_outs: int | None = None,
        in_chans: int = 8,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        attention_type: str = "lrpe",
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        freeze_blocks: bool = False,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.n_layer = n_layer
        self.n_head = n_head
        self.embed_dim = embed_dim

        self.conv_blocks = nn.Sequential(
            ResBlock(in_chans, embed_dim, 2),
            ResBlock(embed_dim, embed_dim, 2),
            ResBlock(embed_dim, embed_dim, 2),
        )
        self.w_raw_in = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [
                CustomAttentionBlock(
                    dim=embed_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    attention_type=attention_type,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(n_layer)
            ]
        )
        self.w_out = nn.Linear(embed_dim, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(embed_dim, num_aux_outs)

    def forward(self, x_feat: torch.Tensor, x_raw: torch.Tensor, session_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x_feat and session_ids are kept for compatibility but unused.
        """
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :].clone()  # shift left r
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)  # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)  # N B D
        x_raw = x_raw.transpose(1, 2)  # B N D
        x_raw = self.w_raw_in(x_raw)  # B N D

        x = x_raw
        for blk in self.blocks:
            x = blk(x)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)

        return self.w_out(x)
