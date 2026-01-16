from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class WaveformAugment(nn.Module):
    def __init__(self, signal_mask_param=200, augment_prob=0.5):
        super().__init__()
        self.signal_mask_param = signal_mask_param
        self.augment_prob = augment_prob

    def forward(self, x):
        batch_size, num_channels, signal_length = x.shape

        # Create a mask for whether to apply augmentation to each channel
        augment_mask = torch.rand(batch_size, num_channels, 1) < self.augment_prob

        # Generate random mask lengths
        mask_lengths = torch.randint(
            1, self.signal_mask_param + 1, (batch_size, num_channels, 1)
        )

        # Generate random start positions
        start_positions = torch.randint(
            0, signal_length + 1 - self.signal_mask_param, (batch_size, num_channels, 1)
        )

        # Create a range tensor
        range_tensor = torch.arange(signal_length).expand(
            batch_size, num_channels, signal_length
        )

        # Create the mask
        mask = (range_tensor < start_positions + mask_lengths) & (
            range_tensor >= start_positions
        )

        # Invert and cast the mask to float
        mask = (~mask).float()

        # Apply the augmentation mask
        mask = torch.where(augment_mask, mask, torch.ones_like(mask))

        # Apply the mask to the input tensor
        return x * mask


class RandomShiftAugment(nn.Module):
    def __init__(self, max_shift=8):
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x):
        batch_size, num_channels, signal_length = x.shape

        r = np.random.randint(self.max_shift)
        if r > 0 and r < signal_length:
            shifted_x = torch.zeros_like(x)
            shifted_x[:, :, :-r] = x[:, :, r:]  # shift left r
            return shifted_x

        return x


class WhiteNoiseAugment(nn.Module):
    def __init__(self, noise_level=0.1, augment_prob=0.5):
        super().__init__()
        self.noise_level = noise_level
        self.augment_prob = augment_prob

    def forward(self, x):
        batch_size, num_channels, signal_length = x.shape

        # Generate white noise
        white_noise = torch.randn_like(x) * self.noise_level

        # Create a mask for whether to apply augmentation to each channel
        augment_mask = (torch.rand(batch_size, num_channels, 1) < self.augment_prob).to(
            x.device
        )

        # Apply the augmentation
        augmented = x + white_noise * augment_mask

        return augmented


class SalientTimeMasking(nn.Module):
    """
    Applies time masking by preferentially selecting high-energy regions.

    Args:
        num_masks (int): The number of masks to apply.
        mask_len (int): The length of each mask in time steps.
        temperature (float): Controls the "peakiness" of the probability distribution.
                             A lower temperature makes masking more biased towards the
                             highest energy peaks. A higher temperature makes it more
                             like uniform random masking.
    """

    def __init__(self, num_masks: int, mask_len: int, temperature: float = 1.0):
        super().__init__()
        self.num_masks = num_masks
        self.mask_len = mask_len
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, Channels)
        batch_size, num_steps, num_channels = x.shape

        # Calculate Saliency (Energy)
        # L2 norm (sum of squares) across the feature dimension.
        # Shape: (Batch, Time)
        energy = torch.sum(x**2, dim=-1)

        # Convert Saliency to Probability
        # softmax with a temperature to control the bias.
        mask_start_probs = F.softmax(energy / self.temperature, dim=-1)

        # Sample Mask Start Positions
        # torch.multinomial samples indices based on the input probabilities.
        # Shape: (Batch, num_masks)
        try:
            mask_start_indices = torch.multinomial(
                mask_start_probs,
                num_samples=self.num_masks,
                replacement=True,  # Allow a mask to start near another mask
            )
        except RuntimeError:
            # can happen if probabilities are all zero (silent input)
            # fall back to random uniform sampling.
            mask_start_indices = torch.randint(
                0, num_steps, (batch_size, self.num_masks), device=x.device
            )

        # Create and Apply the Mask
        x_aug = x.clone()
        for i in range(self.num_masks):
            starts = mask_start_indices[:, i]  # Shape: (Batch,)
            time_range = torch.arange(num_steps, device=x.device).unsqueeze(
                0
            )  # Shape: (1, Time)

            # boolean mask for each item in the batch
            # Shape: (Batch, Time)
            current_mask = (time_range >= starts.unsqueeze(1)) & (
                time_range < (starts + self.mask_len).unsqueeze(1)
            )

            # Apply the mask, values to zero
            x_aug = x_aug.masked_fill(current_mask.unsqueeze(-1), 0.0)

        return x_aug
