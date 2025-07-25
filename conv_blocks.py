from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ResBlock(nn.Module):
    def __init__(
            self, 
            num_ins, 
            num_outs, 
            stride=1,
            act_fn=nn.ReLU,
            norm_layer=nn.BatchNorm1d
        ):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = norm_layer(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = norm_layer(num_outs)
        self.act_fn = act_fn()

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = norm_layer(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = self.act_fn(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return self.act_fn(x + res)

class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module according to https://arxiv.org/pdf/2005.08100
    """
    def __init__(
        self,
        input_size,
        depthwise_kernel_size: int,
        dilation=1,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: nn.Module = nn.LayerNorm,
        act_fn: nn.Module = nn.SiLU,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.padding = (depthwise_kernel_size - 1) * 2 ** (dilation - 1) // 2
        self.layer_norm = layer_norm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=depthwise_kernel_size,
            stride=1,
            padding=self.padding,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            act_fn(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        out = self.layer_norm(input)
        out = rearrange(out, 'b t d -> b d t')
        out = self.bottleneck(out)
        out = self.conv(out)
        out = rearrange(out, 'b d t -> b t d')
        out = self.after_conv(out)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out