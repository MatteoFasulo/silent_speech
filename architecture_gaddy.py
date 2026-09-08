"""Original Gaddy recognition architecture, kept separate from TinyMyo."""

import random

import torch
from torch import nn
import torch.nn.functional as F

from transformer_gaddy import TransformerEncoderLayer


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


class GaddyModel(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None,
                 model_size=768, num_layers=6, dropout=0.2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=model_size, nhead=8, relative_positional=True,
            relative_positional_distance=100, dim_feedforward=3072,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.w_out = nn.Linear(model_size, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(model_size, num_aux_outs)

    def forward(self, x_feat, x_raw, session_ids):
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0
        x_raw = self.conv_blocks(x_raw.transpose(1, 2)).transpose(1, 2)
        x = self.w_raw_in(x_raw).transpose(0, 1)
        x = self.transformer(x).transpose(0, 1)
        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)
        return self.w_out(x)
