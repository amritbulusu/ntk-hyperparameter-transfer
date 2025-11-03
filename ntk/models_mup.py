"""
Neural network architectures using Maximal Update Parametrization (muP).

This uses the official Microsoft mup package for proper muP implementation.
Based on: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
"""

import torch
import torch.nn as nn
from mup import set_base_shapes, MuReadout


class WideMLP_MuP(nn.Module):
    """
    Wide MLP with Maximal Update Parametrization (muP) using the official package.
    
    This network allows feature learning even in the infinite-width limit.
    Key components:
    - Uses mup.MuReadout for output layer
    - Requires calling set_base_shapes for proper muP scaling
    - Works with mup.optim.MuAdam optimizer
    """
    
    def __init__(
        self,
        input_dim: int,
        width: int,
        output_dim: int = 1,
        activation: str = 'relu',
        readout_zero_init: bool = True
    ):
        """
        Args:
            input_dim: Input dimension
            width: Width of the hidden layer (infinite dimension in muP)
            output_dim: Output dimension
            activation: Activation function ('relu' or 'tanh')
            readout_zero_init: If True, initialize output layer to zero
        """
        super(WideMLP_MuP, self).__init__()
        
        self.input_dim = input_dim
        self.width = width
        self.output_dim = output_dim
        self.activation = activation
        
        # Hidden layer: standard Linear layer for infinite x infinite dimensions
        self.fc1 = nn.Linear(input_dim, width, bias=True)
        
        # Output layer: MUST use MuReadout for muP (maps infinite -> finite)
        self.fc2 = MuReadout(width, output_dim, bias=False, readout_zero_init=readout_zero_init)
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
        
        # Note: We use standard PyTorch initialization here
        # set_base_shapes will handle the muP rescaling
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.fc1(x)
        h = self.act(h)
        y = self.fc2(h)
        return y


def make_mup_model(base_width: int = 64, input_dim: int = 1, output_dim: int = 1, 
                   activation: str = 'relu', readout_zero_init: bool = True):
    """
    Create a muP model with proper base shapes setup.
    
    This is a factory function that creates a base model and sets up muP properly.
    
    Args:
        base_width: Base width for the model (used for muP scaling)
        input_dim: Input dimension
        output_dim: Output dimension
        activation: Activation function
        readout_zero_init: Initialize output layer to zero
        
    Returns:
        Model with muP base shapes configured
    """
    model = WideMLP_MuP(input_dim, base_width, output_dim, activation, readout_zero_init)
    return model


def setup_mup_scaling(model, base_width: int = 64):
    """
    Set up muP base shapes for a model.
    
    This function wraps set_base_shapes to provide proper muP scaling.
    Must be called BEFORE creating the optimizer.
    
    Args:
        model: The model to set up (created by make_mup_model)
        base_width: The base width used when creating the model
        
    Returns:
        Model with muP shapes configured
    """
    # Create a temporary base model with the base width
    # to use as reference for set_base_shapes
    temp_base = WideMLP_MuP(model.input_dim, base_width, model.output_dim, 
                            model.activation, False)
    
    # Set base shapes - this adds the infshape attributes and rescales parameters
    set_base_shapes(model, temp_base, rescale_params=True, do_assert=True)
    
    return model
