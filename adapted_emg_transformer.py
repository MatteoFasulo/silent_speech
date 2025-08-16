import sys
import os
import math
import random
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange
from timm.layers import use_fused_attn, trunc_normal_

from transformer import LearnedRelativePositionalEmbedding

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("img_size", 1600, "input image size")
flags.DEFINE_integer("embed_dim", 192, "number of hidden dimensions")
flags.DEFINE_integer("in_chans", 8, "number of input channels")
flags.DEFINE_integer("num_heads", 3, "number of attention heads")
flags.DEFINE_integer("num_layers", 8, "number of layers")
flags.DEFINE_integer("downsample_factor", 8, "downsample factor")
flags.DEFINE_float("mlp_ratio", 4.0, "MLP ratio")
flags.DEFINE_float("dropout", 0.1, "dropout")

class ResBlock(nn.Module):
    def __init__(
        self, num_ins, num_outs, stride=1, pre_activation=False, beta: float = 1.0
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.norm1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_outs)
        # self.act = nn.ReLU()
        self.act = nn.GELU()  # TODO: test which is better
        self.beta = beta

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
            if pre_activation:
                self.skip = nn.Sequential(self.res_norm, self.residual_path)
            else:
                self.skip = nn.Sequential(self.residual_path, self.res_norm)
        else:
            self.skip = nn.Identity()

        # ResNet v2 style pre-activation https://arxiv.org/pdf/1603.05027.pdf
        self.pre_activation = pre_activation

        if pre_activation:
            self.block = nn.Sequential(
                self.norm1, self.act, self.conv1, self.norm2, self.act, self.conv2
            )
        else:
            self.block = nn.Sequential(
                self.conv1, self.norm1, self.act, self.conv2, self.norm2
            )

    def forward(self, x):
        # logging.warning(f"ResBlock forward pass. x.shape: {x.shape}")
        res = self.block(x) * self.beta
        x = self.skip(x)

        if self.pre_activation:
            return x + res
        else:
            return self.act(x + res)

# https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

class LRPEAttention(nn.Module):
    """
    Multi Head Attention with Learned Relative Positional Encoding (LRPE) applied to the logits.
    """
    def __init__(
        self,
        dim,
        num_heads=3,
        qkv_bias=True,
        attn_drop=0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        relative_positional_distance=100,
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

    def forward(self, x, attn_mask=None):
        """Runs the multi-head self-attention layer.

        Args:
          x: the input to the layer, a tensor of shape [batch_size, length, d_model]
        Returns:
          A single tensor containing the output from this layer
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # [B, n_h, N, h_d]
        scale_factor = 1 / math.sqrt(q.size(-1))
        logits = q @ k.transpose(-2, -1) * scale_factor

        # q shape: [B, n_h, N, h_d]
        q_pos = q.permute(0, 2, 1, 3) # [B, N, n_h, h_d]
        b, l, h, d = q_pos.size()
        # The forward pass of relative_positional expects (length, batch*heads, embed_dim)
        position_logits, _ = self.relative_positional(q_pos.reshape(l, b * h, d))
        # position_logits is (b*h, l, l). We need to reshape to (b, h, l, l)
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
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.attn = LRPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            relative_positional_distance=100,
        )
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

    def forward(self, src: torch.Tensor, attn_mask=None) -> torch.Tensor:
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
        num_aux_outs: int = None,
        in_chans: int = 8,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        freeze_blocks: bool = False
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
        # ----------------------------------------------
        self.initialize_weights()

        # Freeze multi-head attention blocks
        if freeze_blocks:
            for param in self.blocks.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        """Initializes the model weights."""
        # Encodings Initializations code taken from the LaBraM paper
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_feat, x_raw, session_ids):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]  # shift left r
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)  # put channel before time for conv
        x_raw = self.conv_blocks(x_raw) # N B D
        x_raw = x_raw.transpose(1, 2) # B N D
        x_raw = self.w_raw_in(x_raw) # B N D

        x = x_raw
        for blk in self.blocks:
            x = blk(x, attn_mask=None)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)

if __name__ == "__main__":
    FLAGS(sys.argv)
    # load pretrained weights if available
    pretrained_path = "/capstor/scratch/cscs/mfasulo/checkpoints/finetuning/epn612/pretrained/full_finetune/run_20250808_111335/checkpoints/new_epn612_pretrained_full_finetune-epoch=39-val_loss=0.4665.ckpt"
    new = EMGTransformer(
        num_features=None,
        num_outs=38,
        num_aux_outs=None,
    )
    print("Model", new)

    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)["state_dict"]
    state_dict = {k.replace('model.','') if k.startswith('model.') else k: v for k, v in state_dict.items()}
    new.load_state_dict(state_dict, strict=False)
    summary(
        new,
        input_size=[(1, FLAGS.img_size, FLAGS.in_chans), (1, FLAGS.img_size, FLAGS.in_chans), (1, FLAGS.img_size, 1)],
        depth=10,
    )
