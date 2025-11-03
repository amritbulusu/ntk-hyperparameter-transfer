"""
Utility functions for NTK experiments including kernel computations,
network initialization, and visualization helpers.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def compute_ntk_kernel(
    model: nn.Module,
    x1: torch.Tensor,
    x2: Optional[torch.Tensor] = None,
    n_samples: int = 100
) -> torch.Tensor:
    """
    Compute the Neural Tangent Kernel (NTK) between two sets of inputs.
    
    The NTK is defined as: K(x1, x2) = sum over all parameters theta of
    (df(x1)/dtheta) · (df(x2)/dtheta)
    
    Args:
        model: Neural network model
        x1: First set of inputs [n1, d]
        x2: Second set of inputs [n2, d]. If None, uses x1.
        n_samples: Not used, kept for API compatibility
        
    Returns:
        Kernel matrix [n1, n2]
    """
    if x2 is None:
        x2 = x1
    
    model.train()  # Need to enable gradients
    kernel = torch.zeros(x1.shape[0], x2.shape[0])
    
    # Compute kernel by computing gradients w.r.t. all parameters
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            # Forward pass for both inputs
            x1_input = x1[i:i+1].detach().requires_grad_(False)
            x2_input = x2[j:j+1].detach().requires_grad_(False)
            
            # Get outputs
            out1 = model(x1_input)
            out2 = model(x2_input)
            
            # Compute gradients w.r.t. all parameters for x1
            grad1_list = torch.autograd.grad(
                outputs=out1.sum(),
                inputs=list(model.parameters()),
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            # Re-forward pass for x2 (need fresh computation)
            out2 = model(x2_input)
            
            # Compute gradients w.r.t. all parameters for x2
            grad2_list = torch.autograd.grad(
                outputs=out2.sum(),
                inputs=list(model.parameters()),
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            
            # Compute inner product of gradient vectors
            k_ij = 0.0
            for grad1, grad2 in zip(grad1_list, grad2_list):
                if grad1 is not None and grad2 is not None:
                    k_ij += (grad1 * grad2).sum().item()
            
            kernel[i, j] = k_ij
    
    return kernel


def compute_ntk_analytical(
    x1: torch.Tensor,
    x2: Optional[torch.Tensor] = None,
    sigma_w: float = 1.0,
    sigma_b: float = 0.0,
    depth: int = 1
) -> torch.Tensor:
    """
    Compute the NTK analytically for a simple ReLU network.
    
    This is a simplified version for demonstration purposes.
    For ReLU networks, the NTK has closed-form expressions.
    
    Args:
        x1: First set of inputs [n1, d]
        x2: Second set of inputs [n2, d]. If None, uses x1.
        sigma_w: Weight variance
        sigma_b: Bias variance
        depth: Network depth
        
    Returns:
        Kernel matrix [n1, n2]
    """
    if x2 is None:
        x2 = x1
    
    # Normalize inputs
    x1_norm = x1 / (x1.norm(dim=1, keepdim=True) + 1e-8)
    x2_norm = x2 / (x2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute dot products
    dot_prods = torch.mm(x1_norm, x2_norm.t())
    
    # For ReLU networks, the kernel has a specific form
    # This is a simplified approximation
    kernel = sigma_w**2 * torch.clamp(dot_prods, min=0.0)
    
    return kernel


def init_weights_gaussian(model: nn.Module, sigma_w: float = 1.0, sigma_b: float = 0.1):
    """
    Initialize network weights as Gaussian random variables.
    
    Args:
        model: Neural network model
        sigma_w: Standard deviation for weights
        sigma_b: Standard deviation for biases
    """
    for param in model.parameters():
        if len(param.shape) >= 2:  # Weight matrix
            nn.init.normal_(param, mean=0.0, std=sigma_w / np.sqrt(param.shape[1]))
        else:  # Bias vector
            nn.init.normal_(param, mean=0.0, std=sigma_b)


def plot_kernel_matrix(kernel: torch.Tensor, title: str = "Kernel Matrix"):
    """
    Plot a kernel matrix as a heatmap.
    
    Args:
        kernel: Kernel matrix [n, n]
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(kernel.detach().cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Kernel Value')
    plt.title(title)
    plt.xlabel('Input Index')
    plt.ylabel('Input Index')
    plt.tight_layout()
    return plt.gcf()


def sample_gp_from_kernel(
    kernel: torch.Tensor,
    n_samples: int = 1,
    noise: float = 1e-6
) -> torch.Tensor:
    """
    Sample functions from a Gaussian Process defined by a kernel matrix.
    
    Args:
        kernel: Kernel matrix [n, n]
        n_samples: Number of function samples
        noise: Small noise for numerical stability
        
    Returns:
        Function samples [n_samples, n]
    """
    n = kernel.shape[0]
    # Add small noise for numerical stability
    kernel_reg = kernel + noise * torch.eye(n, device=kernel.device)
    
    # Sample from multivariate Gaussian
    try:
        L = torch.linalg.cholesky(kernel_reg)
        z = torch.randn(n_samples, n, device=kernel.device)
        samples = torch.mm(z, L.t())
        return samples
    except RuntimeError:
        # If Cholesky fails, use pseudo-inverse
        kernel_reg = kernel + 1e-4 * torch.eye(n, device=kernel.device)
        L = torch.linalg.cholesky(kernel_reg)
        z = torch.randn(n_samples, n, device=kernel.device)
        samples = torch.mm(z, L.t())
        return samples

