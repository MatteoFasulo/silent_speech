import torch
import torch.nn as nn

class FFNModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: nn.Module = nn.LayerNorm,
        act_fn: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            layer_norm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=bias),
            act_fn(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=bias),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)