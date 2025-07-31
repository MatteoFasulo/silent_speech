import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

from timm.layers import SwiGLU
from timm.layers import trunc_normal_ as __call_trunc_normal_

from patch_embedding import PatchEmbedWaveformKeepChans
from attention import CustomAttentionBlock

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('img_size', 1600, 'input image size')
flags.DEFINE_integer('patch_size', 80, 'patch size')
flags.DEFINE_integer('in_chans', 8, 'number of input channels')
flags.DEFINE_integer('embed_dim', 512, 'embedding dimension')
flags.DEFINE_integer('num_heads', 8, 'number of attention heads')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_integer('downsample_factor', 8, 'downsample factor')
flags.DEFINE_float('mlp_ratio', 4.0, 'MLP ratio')
flags.DEFINE_integer('depthwise_conv_kernel_size', 31, 'depthwise convolution kernel size')

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

        # Some checks
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

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
    def __init__(
            self, 
            num_features,
            num_outs: int, 
            num_aux_outs=None, 
            freeze_transformer: bool = False
        ):
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

    def forward(self, x_feat, x_raw, session_ids):
        # x shape: [B, T, C]

        x = x_raw

        x = self.model(x, directly_input_tokens=True) # [B, N, D]

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x) # [B, N, num_outs], [B, N, num_aux_outs]
        else:
            return self.w_out(x) # [B, N, num_outs]

if __name__ == "__main__":
    import sys
    FLAGS(sys.argv)

    #model = NewModel(num_outs=80, num_aux_outs=48)
    model = Conformer(output_dim = 38)
    model.eval()
    lengths = torch.tensor([FLAGS.img_size]) // FLAGS.downsample_factor
    print(model)
    summary(model, 
        input_data=(torch.randn(1, FLAGS.img_size, FLAGS.in_chans), lengths),
    )
    frames, lengths = model(
        torch.randn(1, FLAGS.img_size, FLAGS.in_chans), lengths
    )
    print(frames.shape, lengths)