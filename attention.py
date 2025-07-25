import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import Mlp, LayerScale, DropPath

from pos_embedding import RotaryEmbedding

################
# RoPE Attention
################

class RoPEAttention(nn.Module):
    def __init__(self,
            dim: int,
            num_heads: int,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
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

class PrecomputedRoPESinusoids(nn.Module):
    """
    A cache for the sines and cosines needed to rotate the vectors for rotary
    position embeddings (RoPE).
    This stores the nonzero entries from eq(15) from
    https://arxiv.org/pdf/2104.09864

    Arguments
    ---------
    max_length : int
        The allowed max length of the input sequence.
        For a fixed setting of the other arguments, the computation takes
        O(max_length) time.
    input_size : int
        Size of each vector in the input sequence, i.e. the dimension of each
        attention head.
    dtype : torch.dtype
        The dtype of the tensors.
    device : torch.device
        The Torch device to put the tensors on.

    Example
    -------
    >>> precomputed = PrecomputedRoPESinusoids(3, 8, torch.float32, torch.device('cpu'))
    >>> precomputed.cosines.shape
    torch.Size([3, 8])
    >>> precomputed.sines.shape == precomputed.cosines.shape
    True
    >>> precomputed.cosines
    tensor([[ 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000],
            [ 0.5403,  0.5403,  0.9950,  0.9950,  0.9999,  0.9999,  1.0000,  1.0000],
            [-0.4161, -0.4161,  0.9801,  0.9801,  0.9998,  0.9998,  1.0000,  1.0000]])
    >>> precomputed.sines
    tensor([[-0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000, -0.0000,  0.0000],
            [-0.8415,  0.8415, -0.0998,  0.0998, -0.0100,  0.0100, -0.0010,  0.0010],
            [-0.9093,  0.9093, -0.1987,  0.1987, -0.0200,  0.0200, -0.0020,  0.0020]])
    >>> precomputed.index_swap
    tensor([1, 0, 3, 2, 5, 4, 7, 6])
    """

    def __init__(
        self,
        max_length: int,
        input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        # To precompute the values, use at least float32, because
        # otherwise final accuracy is unnecessarily dreadful.
        internal_dtype = (
            torch.float64 if dtype == torch.float64 else torch.float32
        )

        assert (input_size % 2) == 0

        self.max_length = max_length

        # 10000**(-2(i-1)/d) for i in [1,2,...,d/2]
        angles = torch.exp(
            torch.arange(0, input_size, 2, dtype=internal_dtype, device=device)
            * -(math.log(10000.0) / input_size)
        )

        dimensions = torch.arange(input_size, device=device)

        times = torch.arange(0, max_length, dtype=internal_dtype, device=device)

        # equation (15) without zeros in the matrix
        times_angles = torch.outer(times, angles)

        # Construct
        #     [cos(theta_0), cos(theta_0), cos(theta_1), cos(theta_1), ... ]
        # for equation (34)
        cosines = torch.cos(times_angles)
        cosines = torch.stack([cosines, cosines], dim=-1).reshape(
            max_length, input_size
        )

        # Construct
        #     [sin(theta_0), -sin(theta_0), sin(theta_1), -sin(theta_1), ... ]
        # for equation (34)
        unsigned_sines = torch.sin(times_angles)
        unsigned_repeated_sines = torch.stack(
            [unsigned_sines, unsigned_sines], dim=-1
        ).reshape(max_length, input_size)

        sines = (
            (-1)
            ** torch.arange(input_size, dtype=internal_dtype, device=device)
        ) * -unsigned_repeated_sines

        # To perform a 2-d rotation of every pair of dimensions, a vector will
        # need to be created with every pair swapped with each other.
        # To make this easy, swap every pair of indices:
        # [1, 0, 3, 2, 5, 4, 7, 6, ...]
        index_swap = torch.stack(
            [dimensions[1::2], dimensions[::2]], dim=-1
        ).reshape(-1)

        self.register_buffer("cosines", cosines.to(dtype))
        self.register_buffer("sines", sines.to(dtype))
        self.register_buffer("index_swap", index_swap)


class MemoiseAtLeastSize:
    """
    Memoises a function which has as its first argument a value that indicates a
    minimum value to call the underlying function with.

    Arguments
    ---------
    function: Callable
        The function to call.
    round_up: Callable[[Any], Any]
        A function that rounds up.
        The fewer values this rounds up to, the less likely it is that the
        function will be called repeatedly.
    """

    def __init__(self, function: Callable, round_up: Callable[[Any], Any]):
        self.function = function
        self.round_up = round_up
        # A memo from (parameters 2, 3, ...) to (parameter_1_rounded, result)
        # that stores the result of the call to
        # function(parameter_1_rounded, parameters 2, 3, ...).
        self.memo: Dict[tuple, Tuple[Any, Any]] = {}

    def __call__(self, size: Any, *args):
        if args not in self.memo or self.memo[args][0] < size:
            rounded_size = self.round_up(size)
            assert not (rounded_size < size)
            self.memo[args] = rounded_size, self.function(rounded_size, *args)
        return self.memo[args][1]


def memoise_at_least(
    round_up: Callable[[Any], Any]
) -> Callable[[Callable], MemoiseAtLeastSize]:
    """
    Decorator that memoises a function which has as its first argument a value
    that indicates a minimum value to call the underlying function with.
    If the memo has stored the result from a matching previous function call,
    The stored result will be returned instead of calling the function again.

    Arguments
    ---------
    round_up: Callable[[Any], Any]
        A function that rounds up.
        This will be called with the first argument passed in.
        The underlying function will receive, instead of this first argument,
        the rounded-up version.
        The fewer values this rounds up to, the less likely it is that the
        function will be called repeatedly.

    Returns
    -------
    The passed function but with MemoiseAtLeastSize capability.
    """

    def with_function(function: Callable) -> MemoiseAtLeastSize:
        """
        Set the function to be memoised.
        """
        return MemoiseAtLeastSize(function, round_up)

    return with_function


@memoise_at_least(lambda length: 2 ** int(math.ceil(math.log2(length))))
def _get_precomputed_values(
    length: int, input_size: int, dtype: torch.dtype, device: torch.device
) -> PrecomputedRoPESinusoids:
    """
    Return an object of type PrecomputedRoPESinusoids that is valid for the
    length, input_size, dtype and device.
    Consider a single (input_size, dtype, device), which are usually fixed for
    one model.
    The sinusoids will be recomputed only if they are not yet available for such
    a long length (because of the decorator applied to the function).
    Each time they are precomputed, the length is rounded up to the next power
    of two.

    As a consequence, the total number of calls during one program run is
    upper-bounded by ceil(log2(max_length)) where max_length is the highest
    length that is seen in the program run.
    On realistic lengths, the total number of calls is likely only a few.
    The total number of time steps for which sinusoids are precomputed during
    the program run is O(max_length).

    Arguments
    ---------
    length : int
        The length of the input sequence.
    input_size : int
        Size of each vector in the input sequence, i.e. the dimension of each
        attention head.
    dtype : torch.dtype
        The dtype of the tensors.
    device : torch.device
        The Torch device to put the tensors on.

    Return
    ------
    An object of type PrecomputedRoPESinusoids that is valid for the length,
    input_size, dtype and device.
    """
    # length should have been rounded up to the nearest power of two by the
    # decorator.
    length_power = int(round(math.log2(length)))
    assert length == 2**length_power
    return PrecomputedRoPESinusoids(length, input_size, dtype, device)

def _rope_rotate(x):
    """
    Perform the rotation for RoPE on each of the vectors in x.
    Details about RoPE: https://arxiv.org/pdf/2104.09864.
    """
    _batch_size, length, _num_heads, head_dim = x.shape

    assert (head_dim % 2) == 0

    precomputed = _get_precomputed_values(length, head_dim, x.dtype, x.device)

    # Cut the sinusoids down to the correct length.
    cosines = precomputed.cosines[:length]
    sines = precomputed.sines[:length]

    # The fast implementation for pair-wise rotation requires a version of x
    # with the elements of each pair swapped.
    # (34) in https://arxiv.org/pdf/2104.09864.
    swapped_pairs = torch.index_select(x, dim=-1, index=precomputed.index_swap)

    # (batch_size, L, num_heads, head_dim) * (L, 1, hdead_dim)
    return x * cosines.unsqueeze(1) + swapped_pairs * sines.unsqueeze(1)

class RoPEMHA(nn.Module):
    """This is an implementation of multihead self-attention with RoPE positional embeddings. As it relies on Torch for self-attention, it is
    significantly faster than RelPosMHAXL while offering the same or better levels of accuracy.

    Details about RoPE: https://arxiv.org/pdf/2104.09864.


    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).

    Example
    -------
    >>> max_len = 64
    >>> inputs = torch.rand([6, 60, 512])
    >>> num_heads = 8
    >>> net = RoPEMHA(num_heads=num_heads, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.vhead_dim * num_heads == self.vdim
        ), "vdim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.out_proj = nn.Linear(self.vdim, embed_dim)

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()

        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        pos_embs=None,
        return_attn_weights=True,
    ):
        """Compute attention through Pytorch attention.

        Arguments
        ---------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.BoolTensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.
        pos_embs : torch.Tensor
            Not used by this class. It is kept for compliance.
        return_attn_weights : bool
            Whether to additionally return the attention weights.

        Returns
        -------
        out : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        assert pos_embs is None, "pos_embs is not supported"

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    nn.functional.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = nn.functional.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = nn.functional.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = nn.functional.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        q_rotated = _rope_rotate(query)
        k_rotated = _rope_rotate(key)

        final_masks = masks_union(
            bsz, klen, self.num_heads, attn_mask, key_padding_mask
        )

        x = F.scaled_dot_product_attention(
            query=q_rotated.permute(0, 2, 1, 3),
            key=k_rotated.permute(0, 2, 1, 3),
            value=value.permute(0, 2, 1, 3),
            attn_mask=final_masks,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, None  # out, attn_score
        return out


def masks_union(bsz, klen, num_heads, attn_mask, key_padding_mask):
    """This is an utility function combining standard key_padding_mask and
    attn_mask from SpeechBrain into a single one for scaled_dot_product_attention. This function does not support weighting of the attn_score. Hence, if one wish to use float values as masks, they should not use this function.

    Arguments
    ---------
    bsz : int
        Batch size dimension.
    klen : int
        Time dimension of the key tensor. (Sequence length).
    num_heads : int
        Number of heads of the attention module using these masks.
    attn_mask : torch.BoolTensor
        2D mask (L, S) where L is the target sequence length, S is
        the source sequence length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.
    key_padding_mask : torch.BoolTensor
        (B, S) where B is the batch size, S is the source sequence
        length. The positions with the value of True will be ignored while the position with the value of False will be unchanged.

    Returns
    -------
    out : torch.BoolTensor
        (bsz, num_heads, klen, klen) where False values are masked and True are unmasked (opposite of the input tensors).

    """
    final_mask = None

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, klen).expand(
            bsz, num_heads, klen, klen
        )
        final_mask = key_padding_mask

    if attn_mask is not None:
        attn_mask = attn_mask.view(1, 1, klen, klen).expand(
            bsz, num_heads, klen, klen
        )
        final_mask = attn_mask

    if attn_mask is not None and key_padding_mask is not None:
        final_mask = torch.logical_or(attn_mask, key_padding_mask)

    if final_mask is not None:
        final_mask = torch.logical_not(final_mask)

    return final_mask


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