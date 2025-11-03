"""
Experiment 1: Demonstrating that Neural Networks are Gaussian Processes
in the infinite-width limit.

This script shows how as the width of a neural network increases,
its output distribution converges to a Gaussian Process.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from models import WideMLP
from utils import sample_gp_from_kernel, compute_ntk_analytical, plot_kernel_matrix
from scipy.stats import normaltest


def demonstrate_nn_as_gp():
    """
    Main demonstration: Show that wide neural networks behave like GPs.
    """
    print("=" * 60)
    print("Experiment 1: Neural Networks as Gaussian Processes")
    print("=" * 60)
    
    # Generate test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_test = 50
    input_dim = 1
    
    # Create test points on a grid
    x_test = torch.linspace(-3, 3, n_test).reshape(-1, 1).float()
    
    # Test different widths
    widths = [10, 50, 200, 1000]
    n_samples = 100  # Number of random initializations to sample
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, width in enumerate(widths):
        print(f"\nTesting width = {width}...")
        
        # Sample functions from random initializations
        all_outputs = []
        
        for _ in range(n_samples):
            model = WideMLP(input_dim, width, output_dim=1, activation='relu')
            with torch.no_grad():
                output = model(x_test).squeeze()
            all_outputs.append(output.numpy())
        
        all_outputs = np.array(all_outputs)  # [n_samples, n_test]
        
        # Compute empirical mean and variance
        mean_output = np.mean(all_outputs, axis=0)
        std_output = np.std(all_outputs, axis=0)
        
        # Plot sampled functions
        ax = axes[idx]
        x_test_1d = x_test.numpy().flatten()  # Convert to 1D for matplotlib
        for i in range(min(10, n_samples)):  # Plot first 10 samples
            ax.plot(x_test_1d, all_outputs[i], alpha=0.3, linewidth=0.5, color='blue')
        
        # Plot mean ± 2 std (95% confidence interval)
        ax.plot(x_test_1d, mean_output, 'k-', linewidth=2, label='Mean')
        ax.fill_between(
            x_test_1d,
            mean_output - 2 * std_output,
            mean_output + 2 * std_output,
            alpha=0.3,
            color='gray',
            label='±2 std'
        )
        
        ax.set_title(f'Width = {width}\n(n={n_samples} samples)')
        ax.set_xlabel('Input x')
        ax.set_ylabel('Output f(x)')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
        
        # Test normality at a fixed point (middle of the range)
        mid_idx = n_test // 2
        mid_point_outputs = all_outputs[:, mid_idx]
        
        # Kolmogorov-Smirnov test for normality (simplified check)
        # For large widths, outputs should be approximately normal
        stat, p_value = normaltest(mid_point_outputs)
        print(f"  Normality test at x={x_test[mid_idx].item():.2f}: p={p_value:.4f}")
    
    plt.tight_layout()
    plt.savefig('nn_as_gp.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'nn_as_gp.png'")
    plt.close()
    
    # Now demonstrate the GP kernel perspective
    print("\n" + "=" * 60)
    print("Computing NTK kernel and sampling from corresponding GP...")
    print("=" * 60)
    
    # Compute NTK kernel
    kernel = compute_ntk_analytical(x_test, sigma_w=1.0, sigma_b=0.1)
    
    # Sample from GP defined by this kernel
    gp_samples = sample_gp_from_kernel(kernel, n_samples=10, noise=1e-6)
    
    # Compare with wide network samples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Wide network samples
    x_test_1d = x_test.numpy().flatten()  # Convert to 1D for matplotlib
    wide_model = WideMLP(input_dim, width=2000, output_dim=1, activation='relu')
    wide_outputs = []
    for _ in range(10):
        wide_model = WideMLP(input_dim, width=2000, output_dim=1, activation='relu')
        with torch.no_grad():
            output = wide_model(x_test).squeeze()
        wide_outputs.append(output.numpy())
        ax1.plot(x_test_1d, output.numpy(), alpha=0.6, linewidth=1)
    
    ax1.set_title('Samples from Wide Neural Network\n(Width = 2000)')
    ax1.set_xlabel('Input x')
    ax1.set_ylabel('Output f(x)')
    ax1.grid(True, alpha=0.3)
    
    # GP samples
    for i in range(10):
        ax2.plot(x_test_1d, gp_samples[i].cpu().numpy(), alpha=0.6, linewidth=1)
    
    ax2.set_title('Samples from Gaussian Process\n(Using NTK Kernel)')
    ax2.set_xlabel('Input x')
    ax2.set_ylabel('Output f(x)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_vs_gp_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison figure saved as 'nn_vs_gp_comparison.png'")
    plt.close()
    
    print("\n✓ Experiment 1 Complete!")
    print("\nKey Insight: As width → ∞, the distribution of neural network")
    print("outputs converges to a Gaussian Process with kernel = NTK.")


if __name__ == "__main__":
    demonstrate_nn_as_gp()

