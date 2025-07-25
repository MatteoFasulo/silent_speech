import torch
from torch import nn
from einops import rearrange
from torchinfo import summary

from timm.layers import SwiGLU
from timm.layers import trunc_normal_ as __call_trunc_normal_

from patch_embedding import PatchEmbedWaveformKeepChans
from attention import CustomAttentionBlock
from conv_blocks import ResBlock
from conformer import ConformerLayer
from augmentations import RandomShiftAugment, SalientTimeMasking, WhiteNoiseAugment

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('img_size', 1600, 'input image size')
flags.DEFINE_integer('patch_size', 80, 'patch size')
flags.DEFINE_integer('in_chans', 8, 'number of input channels')
flags.DEFINE_integer('embed_dim', 192, 'embedding dimension')
flags.DEFINE_integer('num_heads', 12, 'number of attention heads')
flags.DEFINE_integer('num_layers', 8, 'number of layers')
flags.DEFINE_integer('downsample_factor', 8, 'downsample factor')
flags.DEFINE_float('mlp_ratio', 4.0, 'MLP ratio')
flags.DEFINE_integer('depthwise_conv_kernel_size', 31, 'depthwise convolution kernel size')

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def print_model_summary(model, input_size: tuple):
    """
    Print a summary of the model.
    """
    summary(model, input_size=input_size)

class EmgTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
        n_layer,
        n_head,
        mlp_ratio,
        qkv_bias,
        qk_norm,
        attn_drop,
        proj_drop,
        drop_path,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.drop_path = drop_path
        # ----------------------------------------------

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.patch_embedding = PatchEmbedWaveformKeepChans(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList([
            CustomAttentionBlock(
                dim=embed_dim,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=drop_path,
                act_layer=nn.SiLU,
                norm_layer=nn.RMSNorm,
                mlp_layer=SwiGLU,
            )
            for i in range(n_layer)
        ])
        self.norm = nn.RMSNorm(embed_dim)
        # ----------------------------------------------
        self.initialize_weights()

        # Some checks
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

    def initialize_weights(self):
        """Initializes the model weights."""
        # Encodings Initializations code taken from the LaBraM paper
        trunc_normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initializes the model weights."""
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, directly_input_tokens=False, attn_mask=None):
        if not directly_input_tokens:
            # patchify first
            x = self.patch_embedding(x)

        # forward pass through transformer blocks
        # attn_mask passed to avoid attending to padded tokens
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        return x # [B, N, D]

class NewModel(nn.Module):
    def __init__(self, num_outs: int, num_aux_outs=None, freeze_transformer: bool = False):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(FLAGS.in_chans, FLAGS.embed_dim, 2),
            ResBlock(FLAGS.embed_dim, FLAGS.embed_dim, 2),
            ResBlock(FLAGS.embed_dim, FLAGS.embed_dim, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.embed_dim, FLAGS.embed_dim)

        self.model = EmgTransformer(
            img_size=FLAGS.img_size,
            patch_size=FLAGS.patch_size,
            in_chans=FLAGS.in_chans,
            embed_dim=FLAGS.embed_dim,
            n_layer=FLAGS.num_layers,
            n_head=FLAGS.num_heads,
            mlp_ratio=FLAGS.mlp_ratio,
            qkv_bias=FLAGS.qkv_bias,
            qk_norm=FLAGS.qk_norm,
            attn_drop=FLAGS.attn_drop,
            proj_drop=FLAGS.proj_drop,
            drop_path=FLAGS.drop_path,
        )
        self.w_out = nn.Linear(FLAGS.embed_dim, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.embed_dim, num_aux_outs)

        if freeze_transformer:
            # freeze the transformer weights
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"Transformer weights frozen")

    def forward(self, x: torch.Tensor):
        # x shape: [B, T, C]

        # Apply data augmentation when training
        if self.training:
            # B C T as expected by the augmentations
            x = rearrange(x, 'b t c -> b c t')
            x = RandomShiftAugment(max_shift=8)(x)
            x = WhiteNoiseAugment(noise_level=0.15, augment_prob=0.5)(x)
            x = rearrange(x, 'b c t -> b t c')  # back to [B, T, C]

        x = rearrange(x, 'b t c -> b c t')
        x = self.conv_blocks(x)  # [B, C, T]
        x = rearrange(x, 'b c t -> b t c')
        x = self.w_raw_in(x)

        x = self.model(x, directly_input_tokens=True) # [B, N, D]

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x) # [B, N, num_outs], [B, N, num_aux_outs]
        else:
            return self.w_out(x) # [B, N, num_outs]

class Conformer(nn.Module):
    def __init__(self, output_dim: int,):
        super().__init__()
        self.embed_dim: int = FLAGS.embed_dim
        self.num_heads: int = FLAGS.num_heads
        self.ffn_dim: int = int(FLAGS.embed_dim * FLAGS.mlp_ratio)
        self.num_layers: int = FLAGS.num_layers
        self.depthwise_conv_kernel_size: int = FLAGS.depthwise_conv_kernel_size
        self.conv_drop: float = 0.2
        self.attn_drop: float = 0.0
        self.proj_drop: float = 0.2
        self.ffn_dropout: float = 0.2
        self.bias: bool = True
        self.layer_norm: nn.Module = nn.LayerNorm
        self.act_fn: nn.Module = nn.SiLU

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

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    self.embed_dim,
                    self.ffn_dim,
                    self.num_heads,
                    self.depthwise_conv_kernel_size,
                    conv_drop=self.conv_drop,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                    ffn_dropout=self.ffn_dropout,
                    bias=self.bias,
                    layer_norm=self.layer_norm,
                    act_fn=self.act_fn,
                )
                for _ in range(self.num_layers)
            ]
        )
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
        encoder_padding_mask = self._lengths_to_padding_mask(lengths)

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
        x = self.conv_subsampling(x)  # [B, C, T/4]
        x = rearrange(x, 'b c t -> b t c')
        x = self.conv_linear(x)
        x = self.conv_dropout(x)

        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        
        # Final output projection
        logits = self.w_out(x)

        return logits, lengths

if __name__ == "__main__":
    import sys
    FLAGS(sys.argv)

    #model = NewModel(num_outs=80, num_aux_outs=48)
    model = Conformer(output_dim = 38)
    model.eval()
    lengths = torch.tensor([1600]) // FLAGS.downsample_factor
    print(model)
    summary(model, 
        input_data=(torch.randn(1, FLAGS.img_size, FLAGS.in_chans), lengths),
    )
    frames, lengths = model(
        torch.randn(1, FLAGS.img_size, FLAGS.in_chans), lengths
    )
    print(frames.shape, lengths)