"""
Experiment 2: Exploring Neural Tangent Kernel Properties

This script demonstrates:
- How to compute the NTK
- Key properties of the NTK
- How the NTK relates to training dynamics
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import WideMLP
from utils import compute_ntk_analytical, plot_kernel_matrix


def compute_ntk_empirical(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Compute NTK empirically using automatic differentiation.
    
    The NTK is: K(x_i, x_j) = sum_{params} ∇_θ f(x_i) · ∇_θ f(x_j)
    """
    model.eval()
    n = x.shape[0]
    kernel = torch.zeros(n, n)
    
    for i in range(n):
        for j in range(i, n):
            x_i = x[i:i+1].requires_grad_(True)
            x_j = x[j:j+1].requires_grad_(True)
            
            # Forward pass
            f_i = model(x_i)
            f_j = model(x_j)
            
            # Compute gradients for all parameters
            params = list(model.parameters())
            
            # Gradient of f_i w.r.t. parameters
            grad_i = torch.autograd.grad(
                f_i, params, create_graph=False, retain_graph=True, allow_unused=True
            )
            
            # Gradient of f_j w.r.t. parameters
            grad_j = torch.autograd.grad(
                f_j, params, create_graph=False, retain_graph=True, allow_unused=True
            )
            
            # Compute inner product
            k_ij = 0.0
            for g_i, g_j in zip(grad_i, grad_j):
                if g_i is not None and g_j is not None:
                    k_ij += (g_i * g_j).sum().item()
            
            kernel[i, j] = k_ij
            if i != j:
                kernel[j, i] = k_ij
    
    return kernel


def demonstrate_ntk_properties():
    """
    Main demonstration of NTK properties.
    """
    print("=" * 60)
    print("Experiment 2: Neural Tangent Kernel Properties")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate test data
    n = 30
    input_dim = 2
    
    # Create a grid of test points
    x1 = torch.linspace(-2, 2, int(np.sqrt(n)))
    x2 = torch.linspace(-2, 2, int(np.sqrt(n)))
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_test = torch.stack([X1.flatten(), X2.flatten()], dim=1).float()
    n = x_test.shape[0]
    
    print(f"\nTesting with {n} input points...")
    
    # Test different widths
    widths = [50, 200, 1000]
    
    fig, axes = plt.subplots(1, len(widths), figsize=(15, 5))
    
    for idx, width in enumerate(widths):
        print(f"\nWidth = {width}:")
        
        # Create model
        model = WideMLP(input_dim, width, output_dim=1, activation='relu')
        
        # Compute NTK
        print("  Computing NTK kernel...")
        kernel = compute_ntk_empirical(model, x_test)
        
        # Visualize kernel matrix
        ax = axes[idx]
        im = ax.imshow(kernel.detach().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'NTK Matrix (Width = {width})')
        ax.set_xlabel('Input Index')
        ax.set_ylabel('Input Index')
        plt.colorbar(im, ax=ax)
        
        # Check properties
        # 1. Symmetry
        is_symmetric = torch.allclose(kernel, kernel.t(), atol=1e-4)
        print(f"  Symmetry: {is_symmetric}")
        
        # 2. Positive semi-definiteness (check eigenvalues)
        eigenvals = torch.linalg.eigvalsh(kernel)
        min_eigenval = eigenvals.min().item()
        is_psd = min_eigenval >= -1e-5
        print(f"  Positive semi-definite: {is_psd} (min eigenvalue: {min_eigenval:.6f})")
        
        # 3. Kernel values
        print(f"  Kernel range: [{kernel.min().item():.4f}, {kernel.max().item():.4f}]")
        print(f"  Diagonal mean: {kernel.diag().mean().item():.4f}")
    
    plt.tight_layout()
    plt.savefig('ntk_properties.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'ntk_properties.png'")
    plt.close()
    
    # Demonstrate NTK invariance with width scaling
    print("\n" + "=" * 60)
    print("Demonstrating NTK convergence as width increases...")
    print("=" * 60)
    
    # Use simpler 1D inputs for this
    n_simple = 20
    x_simple = torch.linspace(-2, 2, n_simple).reshape(-1, 1).float()
    
    widths_seq = [10, 50, 100, 500, 1000]
    kernels_seq = []
    
    for width in widths_seq:
        model = WideMLP(1, width, output_dim=1, activation='relu')
        kernel = compute_ntk_empirical(model, x_simple)
        kernels_seq.append(kernel)
        print(f"Width {width:4d}: Kernel norm = {kernel.norm().item():.4f}")
    
    # Compare kernels (normalized)
    if len(kernels_seq) >= 2:
        k1 = kernels_seq[0]
        k_last = kernels_seq[-1]
        
        # Normalize for comparison
        k1_norm = k1 / k1.norm()
        k_last_norm = k_last / k_last.norm()
        
        diff = (k1_norm - k_last_norm).norm().item()
        print(f"\nKernel convergence (normalized diff): {diff:.6f}")
        print("(Lower values indicate convergence as width increases)")
    
    # Visualize kernel convergence
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (width, kernel) in enumerate(zip(widths_seq, kernels_seq)):
        if idx < len(axes):
            ax = axes[idx]
            im = ax.imshow(kernel.detach().numpy(), cmap='viridis', aspect='auto')
            ax.set_title(f'Width = {width}')
            plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for idx in range(len(widths_seq), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('ntk_convergence.png', dpi=150, bbox_inches='tight')
    print("Convergence figure saved as 'ntk_convergence.png'")
    plt.close()
    
    print("\n✓ Experiment 2 Complete!")
    print("\nKey Insights:")
    print("1. NTK is symmetric and positive semi-definite (kernel properties)")
    print("2. NTK converges as network width increases")
    print("3. NTK determines training dynamics: df/dt = -η * K * (f - y)")


if __name__ == "__main__":
    demonstrate_ntk_properties()

