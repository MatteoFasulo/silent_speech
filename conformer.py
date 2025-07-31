
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from attention import RelPositionMultiHeadedAttention

class FFNModule(nn.Module):
    def __init__(
        self,
        input_feat: int,
        hidden_units: int,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        activation: nn.Module = nn.GELU,
        layer_norm: nn.Module = nn.LayerNorm,
        bias: bool = True,
    ):
        super().__init__()
        self.layer_norm = layer_norm(input_feat)
        self.w_1 = torch.nn.Linear(input_feat, hidden_units, bias=bias)
        self.w_2 = torch.nn.Linear(hidden_units, input_feat, bias=bias)
        self.dropout1 = torch.nn.Dropout(dropout1)
        self.dropout2 = torch.nn.Dropout(dropout2)
        self.activation = activation()

    def forward(self, x):
        # T B C
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.w_2(x)
        return self.dropout2(x)

class ConvolutionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.SiLU,
        bias: bool = False
    ):
        super().__init__()
        assert (
            depthwise_kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = torch.nn.Conv1d(
            embed_dim,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.glu = torch.nn.GLU(dim=1)
        self.depthwise_conv = torch.nn.Conv1d(
            channels,
            channels,
            depthwise_kernel_size,
            stride=1,
            padding=(depthwise_kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels)
        self.activation = activation(channels)
        self.pointwise_conv2 = torch.nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # B T C
        x = self.layer_norm(x)
        x = rearrange(x, 'b t c -> b c t')
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return rearrange(x, 'b c t -> b t c')

class ConformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        attention_heads: int,
        dropout: float,
        depthwise_conv_kernel_size: int,
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()

        self.ffn1 = FFNModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = RelPositionMultiHeadedAttention(
            embed_dim,
            attention_heads,
            dropout=dropout,
        )

        self.conv_module = ConvolutionModule(
            embed_dim=embed_dim,
            channels=embed_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            activation=activation,
        )

        self.ffn2 = FFNModule(
            embed_dim,
            ffn_embed_dim,
            dropout,
            dropout,
            activation=activation,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        position_emb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Tensor of shape T X B X C
            encoder_padding_mask: Optional mask tensor
            positions:
        Returns:
            Tensor of shape T X B X C
        """
        residual = x
        x = self.ffn1(x)
        x = x * 0.5 + residual
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            pos_emb=position_emb,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        residual = x
        # TBC to BTC
        x = rearrange(x, 't b c -> b t c')
        x = self.conv_module(x)
        # BTC to TBC
        x = rearrange(x, 'b t c -> t b c')
        x = residual + x

        residual = x
        x = self.ffn2(x)

        layer_result = x

        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x

class Conformer(nn.Module):
    def __init__(self, output_dim: int,):
        super().__init__()
        self.embed_dim: int = FLAGS.embed_dim
        self.num_heads: int = FLAGS.num_heads
        self.ffn_dim: int = int(FLAGS.embed_dim * FLAGS.mlp_ratio)
        self.num_layers: int = FLAGS.num_layers
        self.depthwise_conv_kernel_size: int = FLAGS.depthwise_conv_kernel_size
        self.conv_drop: float = 0.2
        self.attn_drop: float = 0.2
        self.bias: bool = True
        self.layer_norm: nn.Module = nn.LayerNorm
        self.act_fn: nn.Module = nn.SiLU

        self.embed_scale = math.sqrt(FLAGS.embed_dim)
        self.embed_positions = RelPositionalEncoding(
            200,  # max_len
            FLAGS.embed_dim
        )

        self.conv_subsampling = nn.Sequential(
            # Downsample 2x (e.g., from 1600 to 800)
            ResBlock(FLAGS.in_chans, self.embed_dim, 2),
            # Downsample 2x (e.g., from 800 to 400)
            ResBlock(self.embed_dim, self.embed_dim, 2),
            # Downsample 2x (e.g., from 400 to 200)
            ResBlock(self.embed_dim, self.embed_dim, 2),
        )
        self.conv_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.conv_dropout = nn.Dropout(self.conv_drop)

        self.layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=self.ffn_dim,
                    attention_heads=self.num_heads,
                    dropout=self.attn_drop,
                    depthwise_conv_kernel_size=self.depthwise_conv_kernel_size,
                    activation=self.act_fn,
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = self.layer_norm(self.embed_dim, eps=1e-6)
        self.w_out = nn.Linear(self.embed_dim, output_dim)

        self.waveform_augmentations = nn.Sequential(
            RandomShiftAugment(max_shift=8),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes the model weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _lengths_to_padding_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_length = int(torch.max(lengths).item())
        padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
            batch_size, max_length
        ) >= lengths.unsqueeze(1)
        return padding_mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [B, T, C]
        padding_mask = self._lengths_to_padding_mask(lengths)

        # Apply data augmentation when training
        if self.training:
            # B C T as expected by the augmentations
            x = rearrange(x, 'b t c -> b c t')
            x = self.waveform_augmentations(x)
            x = rearrange(x, 'b c t -> b t c')  # back to [B, T, C]

        # Conv stemming
        # 1) Downsampling
        # 2) Linear
        # 3) Dropout
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv_subsampling(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.conv_linear(x)
        x = self.conv_dropout(x)

        x = rearrange(x, 'b t c -> t b c')
        x = self.embed_scale * x
        positions = self.embed_positions(x)  # Add positional encodings

        for layer in self.layers:
            x = layer(x, padding_mask, positions)
        x = rearrange(x, 't b c -> b t c')

        x = self.norm(x)

        # Final output projection
        logits = self.w_out(x)

        return logits, lengths