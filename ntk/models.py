"""
Neural network architectures for NTK experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class MLP(nn.Module):
    """
    Multi-layer perceptron for NTK experiments.
    Can be made arbitrarily wide to approach the infinite-width limit.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: str = 'relu',
        bias: bool = True
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer widths
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh', 'linear')
            bias: Whether to use bias terms
        """
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.bias = bias
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            if i < len(dims) - 2:  # No activation after last layer
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                # 'linear' means no activation
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def get_width(self) -> int:
        """Get the width of the first hidden layer."""
        return self.hidden_dims[0] if len(self.hidden_dims) > 0 else 0


class WideMLP(nn.Module):
    """
    Simplified wide MLP for studying infinite-width behavior.
    Single hidden layer with variable width.
    """
    
    def __init__(
        self,
        input_dim: int,
        width: int,
        output_dim: int = 1,
        activation: str = 'relu',
        sigma_w: float = 1.0,
        sigma_b: float = 0.1
    ):
        """
        Args:
            input_dim: Input dimension
            width: Width of the hidden layer
            output_dim: Output dimension
            activation: Activation function
            sigma_w: Weight initialization standard deviation
            sigma_b: Bias initialization standard deviation
        """
        super(WideMLP, self).__init__()
        
        self.input_dim = input_dim
        self.width = width
        self.output_dim = output_dim
        self.activation = activation
        
        self.fc1 = nn.Linear(input_dim, width, bias=True)
        self.fc2 = nn.Linear(width, output_dim, bias=False)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        
        # Initialize with proper scaling for infinite-width limit
        nn.init.normal_(self.fc1.weight, mean=0.0, std=sigma_w / np.sqrt(input_dim))
        if self.fc1.bias is not None:
            nn.init.normal_(self.fc1.bias, mean=0.0, std=sigma_b)
        
        nn.init.normal_(self.fc2.weight, mean=0.0, std=sigma_w / np.sqrt(width))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.fc1(x)
        h = self.act(h)
        y = self.fc2(h)
        return y

