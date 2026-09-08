"""Transformer components used by Gaddy's original silent-speech model."""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class LearnedRelativePositionalEmbedding(nn.Module):
    """Learned relative position logits, matching Gaddy's implementation."""

    def __init__(self, max_relative_pos, num_heads, embedding_dim, unmasked=False,
                 heads_share_embeddings=False, add_to_values=False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = 2 * max_relative_pos - 1 if unmasked else max_relative_pos
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=embedding_dim ** -0.5)

    def forward(self, query, saved_state=None):
        # Sequence-first contract: query is [L, B * H, D].
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked
            length = saved_state["prev_key"].shape[-2] + 1
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used = self.get_embeddings_for_query(length)
        values = used[..., 1] if self.add_to_values else None
        logits = self.calculate_positional_logits(query, used[..., 0])
        return self.relative_to_absolute_indexing(logits, decoder_step), values

    def get_embeddings_for_query(self, length):
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded = F.pad(self.embeddings, (0, 0, 0, 0, pad_length, pad_length))
            return padded.narrow(-3, start_pos, 2 * length - 1)
        with torch.no_grad():
            padded = F.pad(self.embeddings, (0, 0, 0, 0, pad_length, 0))
        return padded.narrow(-3, start_pos, length)

    def calculate_positional_logits(self, query, relative_embeddings):
        if self.heads_share_embeddings:
            logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            logits = logits.contiguous().view(logits.shape[0], -1, logits.shape[-1])
        length = query.size(0)
        if length > self.max_relative_pos:
            pad_length = length - self.max_relative_pos
            logits[:, :, :pad_length] -= 1e8
            if self.unmasked:
                logits[:, :, -pad_length:] -= 1e8
        return logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        length, bsz_heads, _ = x.shape
        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)
        if self.unmasked:
            x = F.pad(x, (0, 1)).transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = F.pad(x, (0, length - 1))
            x = x.view(bsz_heads, length + 1, 2 * length - 1)
            return x[:, :length, length - 1:]
        x = F.pad(x, (1, 0)).transpose(0, 1)
        x = x.contiguous().view(bsz_heads, length + 1, length)
        return x[:, 1:, :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_head=4, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_qkv = d_model // n_head
        # PyTorch's TransformerEncoder fast-path inspection expects this
        # attribute on custom attention modules. Gaddy's implementation uses
        # sequence-first tensors, so this must remain False.
        self.batch_first = False
        assert self.d_qkv * n_head == d_model, "d_model must be divisible by n_head"
        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, self.d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, self.d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        self.dropout = nn.Dropout(dropout)
        self.relative_positional = (
            LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, self.d_qkv, True)
            if relative_positional else None
        )

    def forward(self, x):
        # Gaddy's transformer uses [L, B, D] throughout this layer.
        q = torch.einsum("tbf,hfa->bhta", x, self.w_q)
        k = torch.einsum("tbf,hfa->bhta", x, self.w_k)
        v = torch.einsum("tbf,hfa->bhta", x, self.w_v)
        logits = torch.einsum("bhqa,bhka->bhqk", q, k) / math.sqrt(self.d_qkv)
        if self.relative_positional is not None:
            q_pos = q.permute(2, 0, 1, 3)
            length, batch, heads, depth = q_pos.shape
            pos, _ = self.relative_positional(q_pos.reshape(length, batch * heads, depth))
            assert pos.shape == (batch * heads, length, length)
            logits = logits + pos.view(batch, heads, length, length)
        probs = self.dropout(F.softmax(logits, dim=-1))
        out = torch.einsum("bhqk,bhka->bhqa", probs, v)
        return torch.einsum("bhta,haf->tbf", out, self.w_o)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout,
            relative_positional=relative_positional,
            relative_positional_distance=relative_positional_distance,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        src2 = self.self_attn(src)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))
