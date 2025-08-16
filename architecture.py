import sys
import random
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

from transformer import TransformerEncoderLayer

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("img_size", 1600, "input image size")
flags.DEFINE_integer("embed_dim", 768, "number of hidden dimensions")
flags.DEFINE_integer("in_chans", 8, "number of input channels")
flags.DEFINE_integer("num_heads", 8, "number of attention heads")
flags.DEFINE_integer("num_layers", 6, "number of layers")
flags.DEFINE_integer("downsample_factor", 8, "downsample factor")
flags.DEFINE_float("mlp_ratio", 4.0, "MLP ratio")
flags.DEFINE_float("dropout", 0.2, "dropout")


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None, *args, **kwargs):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.embed_dim, 2),
            ResBlock(FLAGS.embed_dim, FLAGS.embed_dim, 2),
            ResBlock(FLAGS.embed_dim, FLAGS.embed_dim, 2),
        )
        self.w_raw_in = nn.Linear(FLAGS.embed_dim, FLAGS.embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=FLAGS.embed_dim,
            nhead=FLAGS.num_heads,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=int(FLAGS.embed_dim * FLAGS.mlp_ratio),
            dropout=FLAGS.dropout,
            batch_first=False,  # [T, B, C] input format to transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.w_out = nn.Linear(FLAGS.embed_dim, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.embed_dim, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        # x shape is (batch, time, electrode)

        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]  # shift left r
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)  # put channel before time for conv
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw

        x = x.transpose(0, 1)  # put time first
        x = self.transformer(x)
        x = x.transpose(0, 1)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        else:
            return self.w_out(x)


if __name__ == "__main__":
    FLAGS(sys.argv)
    model = Model(
        num_features=None,
        num_outs=38,
        num_aux_outs=None,
    )
    model.eval()
    x = torch.randn(1, FLAGS.img_size, FLAGS.in_chans)
    lengths = torch.tensor([FLAGS.img_size]) // FLAGS.downsample_factor
    print(model)
    summary(
        model,
        input_size=[(1, FLAGS.img_size, FLAGS.in_chans), (1, FLAGS.img_size, FLAGS.in_chans), (1, FLAGS.img_size, 1)],
        depth=5,
    )
