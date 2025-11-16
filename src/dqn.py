import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """Robust Double DQN architecture with batch normalization and deeper layers.

    Expects input_shape as (C, H, W) and returns Q-values for n_actions.
    Uses batch normalization for stable training and deeper conv layers.
    """

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Enhanced convolutional feature extractor with batch normalization
        self.conv = nn.Sequential(
            # First conv block
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Fourth conv block (additional depth)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Adaptive pooling for consistent output size
            nn.AdaptiveAvgPool2d((7, 7)),
        )

        conv_out_size = self._get_conv_output(input_shape)

        # Optimized MLP head - reduced size for memory efficiency
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, n_actions),
        )

    def _get_conv_output(self, shape):
        """Calculate the flattened conv output size."""
        with torch.no_grad():
            dummy = torch.zeros(1, *shape)
            out = self.conv(dummy)
            return int(np.prod(out.size()))

    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
