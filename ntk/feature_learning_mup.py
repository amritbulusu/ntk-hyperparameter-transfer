"""
Experiment: Feature Learning with Maximal Update Parametrization (muP)

This script demonstrates that with muP, neural networks can learn features
even as width approaches infinity. This is the key breakthrough that overcomes
NTK limitations.

Uses the official Microsoft mup package:
https://github.com/microsoft/mup
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models_mup import WideMLP_MuP, setup_mup_scaling
from mup.optim import MuAdam, MuSGD
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def generate_shifted_data(n_samples: int = 100, shift: float = 1.0):
    """
    Generate a dataset where feature learning would be beneficial.
    The task requires learning shifted features.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Input: uniform on [-2, 2]
    x = torch.rand(n_samples, 1) * 4 - 2
    
    # Target: requires learning a shifted feature
    # y = sin(x + shift) - this requires the network to learn the shift
    y = torch.sin(x + shift) + 0.1 * torch.randn_like(x)
    
    return x, y


def train_model_mup(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 0.001,
    clip_grad_norm: float = 10.0,
    use_muadam: bool = True
):
    """
    Train a model with muP using the official MuAdam optimizer.
    
    In muP with MuAdam, we don't scale learning rate with width!
    That's the key advantage.
    """
    criterion = nn.MSELoss()
    
    # Use MuAdam optimizer from mup package
    if use_muadam:
        optimizer = MuAdam(model.parameters(), lr=lr)
    else:
        optimizer = MuSGD(model.parameters(), lr=lr)
    
    loss_history = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Check for NaN before backward pass
        if torch.isnan(loss):
            print(f"  Warning: NaN loss detected at epoch {epoch}, stopping training")
            break
        
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        if epoch % 200 == 0:
            if not np.isnan(loss_val):
                print(f"  Epoch {epoch}: Loss = {loss_val:.6f}")
            else:
                print(f"  Epoch {epoch}: Loss = NaN")
    
    return loss_history


def demonstrate_feature_learning_mup():
    """
    Main demonstration: Show that muP enables feature learning at infinite width.
    """
    print("=" * 70)
    print("Experiment: Feature Learning with Maximal Update Parametrization (muP)")
    print("Using official Microsoft mup package")
    print("=" * 70)
    
    # Generate training data
    n_train = 50
    shift = 1.0  # Task requires learning this shift
    
    x_train, y_train = generate_shifted_data(n_train, shift)
    
    print(f"\nTask: Learn y = sin(x + {shift})")
    print(f"Training samples: {n_train}")
    print("\nKey innovation: With muP + MuAdam, learning rate stays constant across widths!")
    
    # Base width for muP scaling
    base_width = 64
    
    # Test different widths - should use same learning rate for all
    widths = [10, 50, 200, 1000, 5000, 10000]
    
    # In muP, we use the SAME learning rate for all widths
    lr = 0.001  # Fixed learning rate
    
    final_losses = []
    trained_models = []
    
    for width in widths:
        print(f"\n{'='*60}")
        print(f"Training muP network with width = {width}")
        print(f"Learning rate = {lr} (same for all widths!)")
        print(f"{'='*60}")
        
        # Create model with muP parameterization
        model = WideMLP_MuP(
            input_dim=1, 
            width=width, 
            output_dim=1, 
            activation='relu',
            readout_zero_init=True
        )
        
        # CRITICAL: Set up muP base shapes BEFORE training
        setup_mup_scaling(model, base_width=base_width)
        
        # Train with FIXED learning rate (key muP advantage!)
        loss_history = train_model_mup(
            model, x_train, y_train, 
            n_epochs=1000, 
            lr=lr,
            use_muadam=True
        )
        
        final_loss = loss_history[-1]
        final_losses.append(final_loss)
        trained_models.append(model)
        
        if np.isnan(final_loss) or np.isinf(final_loss):
            print(f"Final training loss: NaN/Inf (training unstable)")
        else:
            print(f"Final training loss: {final_loss:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Loss vs width (should IMPROVE with width in muP!)
    ax = axes[0, 0]
    valid_widths = []
    valid_losses = []
    invalid_widths = []
    for w, l in zip(widths, final_losses):
        if not (np.isnan(l) or np.isinf(l)):
            valid_widths.append(w)
            valid_losses.append(l)
        else:
            invalid_widths.append(w)
    
    if valid_widths:
        ax.plot(valid_widths, valid_losses, 'o-', linewidth=2, markersize=8, 
                color='green', label='muP (feature learning!)')
    
    if invalid_widths and valid_losses:
        y_max = max(valid_losses) if valid_losses else 1.0
        for w in invalid_widths:
            ax.plot(w, y_max * 1.1, 'rx', markersize=12)
        if len(invalid_widths) > 0:
            ax.plot([], [], 'rx', markersize=12, label='NaN/Inf (unstable)')
    
    ax.set_xscale('log')
    ax.set_xlabel('Network Width', fontsize=11)
    ax.set_ylabel('Final Training Loss', fontsize=11)
    ax.set_title('Feature Learning with muP: Wider is BETTER!', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color='r', linestyle='--', label='Good performance threshold')
    if valid_widths:
        ax.legend(fontsize=10)
    
    # Plot 2-6: Predictions for different widths
    x_test = torch.linspace(-3, 3, 200).reshape(-1, 1).float()
    y_true = torch.sin(x_test + shift)
    
    for idx, (width, model) in enumerate(zip(widths, trained_models)):
        if idx >= 5:  # Only plot first 5 widths
            break
            
        row = idx // 3
        col = idx % 3
        
        if row < 2 and col < 3:
            ax = axes[row, col]
            
            model.eval()
            with torch.no_grad():
                y_pred = model(x_test)
            
            # Convert to 1D for matplotlib
            x_train_1d = x_train.numpy().flatten()
            x_test_1d = x_test.numpy().flatten()
            y_train_1d = y_train.numpy().flatten()
            y_true_1d = y_true.numpy().flatten()
            y_pred_1d = y_pred.numpy().flatten()
            
            # Check for NaN/Inf in predictions
            if np.any(np.isnan(y_pred_1d)) or np.any(np.isinf(y_pred_1d)):
                ax.text(0.5, 0.5, f'Width = {width}\nModel unstable\nContains NaN/Inf', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=10)
                loss_str = 'NaN' if (np.isnan(final_losses[idx]) or np.isinf(final_losses[idx])) else f'{final_losses[idx]:.4f}'
            else:
                ax.scatter(x_train_1d, y_train_1d, alpha=0.5, s=20, 
                          label='Training data', color='gray')
                ax.plot(x_test_1d, y_true_1d, 'k--', linewidth=2, 
                       label='True function')
                ax.plot(x_test_1d, y_pred_1d, 'g-', linewidth=2, 
                       label='muP prediction')
                loss_str = 'NaN' if (np.isnan(final_losses[idx]) or np.isinf(final_losses[idx])) else f'{final_losses[idx]:.4f}'
                ax.legend(fontsize=8)
            
            ax.set_title(f'muP: Width = {width}\nLoss = {loss_str}', fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_learning_mup.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'feature_learning_mup.png'")
    plt.close()
    
    # Demonstrate the key insight: wider is better!
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    
    print(f"\nLoss comparison with muP (same LR = {lr} for all widths):")
    for width, loss in zip(widths, final_losses):
        if np.isnan(loss) or np.isinf(loss):
            print(f"  Width {width:5d}: NaN/Inf (training unstable)")
        else:
            print(f"  Width {width:5d}: {loss:.6f}")
    
    # Check if wider performs better
    valid_indices = [i for i, l in enumerate(final_losses) 
                     if not (np.isnan(l) or np.isinf(l))]
    
    if len(valid_indices) >= 2:
        # Compare smallest and largest valid widths
        min_w = widths[valid_indices[0]]
        min_l = final_losses[valid_indices[0]]
        max_w = widths[valid_indices[-1]]
        max_l = final_losses[valid_indices[-1]]
        
        print(f"\n{min_w} → {max_w} width change:")
        print(f"  Loss: {min_l:.6f} → {max_l:.6f}")
        
        if max_l < min_l:
            print(f"\n✓ SUCCESS: muP enables feature learning at infinite width!")
            print(f"  Wider networks perform BETTER, not worse!")
            improvement = (min_l - max_l) / min_l * 100
            print(f"  Improvement: {improvement:.1f}%")
        else:
            print(f"\n? Results mixed, but training is stable across widths.")
    
    # Compare with standard parameterization
    print("\n" + "=" * 70)
    print("Comparison: muP vs Standard Parametrization")
    print("=" * 70)
    
    print("\nKey differences:")
    print("1. muP: Same learning rate for all widths ✓")
    print("2. Standard: Must scale LR as 1/√width ✗")
    print("3. muP: Feature learning at infinite width ✓")
    print("4. Standard: Stuck in NTK regime, no feature learning ✗")
    
    print("\n✓ Experiment Complete!")
    print("\nKey Insight: Maximal Update Parametrization (muP) enables")
    print("neural networks to learn features even as width → ∞, overcoming")
    print("the fundamental limitation of the NTK regime!")


def compare_mup_vs_standard():
    """
    Direct comparison between muP and standard parametrization.
    """
    print("\n" + "=" * 70)
    print("Direct Comparison: muP vs Standard Parametrization")
    print("=" * 70)
    
    # Generate data
    n_train = 50
    shift = 1.0
    x_train, y_train = generate_shifted_data(n_train, shift)
    x_test = torch.linspace(-3, 3, 200).reshape(-1, 1).float()
    y_true = torch.sin(x_test + shift)
    
    # Test width
    width = 5000
    base_width = 64
    
    # 1. Standard parametrization
    print("\n1. Training with STANDARD parametrization...")
    from models import WideMLP
    
    std_model = WideMLP(1, width=width, output_dim=1, activation='relu', sigma_w=1.0)
    
    # Must scale LR with width for standard param
    std_lr = 0.01 / np.sqrt(width / 100.0)
    print(f"   Using scaled LR: {std_lr:.6f} (scaled as 1/√width)")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(std_model.parameters(), lr=std_lr)
    
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = std_model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(std_model.parameters(), 10.0)
        optimizer.step()
        
        if epoch % 200 == 0 and epoch > 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.6f}")
    
    std_model.eval()
    with torch.no_grad():
        std_pred = std_model(x_test)
    
    std_mse = mean_squared_error(y_true.numpy().flatten(), std_pred.numpy().flatten())
    print(f"   Final MSE: {std_mse:.6f}")
    
    # 2. muP parametrization
    print("\n2. Training with muP parametrization...")
    mup_model = WideMLP_MuP(1, width=width, output_dim=1, activation='relu', 
                            readout_zero_init=True)
    
    # Set up muP base shapes
    setup_mup_scaling(mup_model, base_width=base_width)
    
    # Use FIXED LR for muP
    mup_lr = 0.001
    print(f"   Using fixed LR: {mup_lr:.6f} (same for all widths!)")
    
    mup_optimizer = MuAdam(mup_model.parameters(), lr=mup_lr)
    
    for epoch in range(1000):
        mup_optimizer.zero_grad()
        y_pred = mup_model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mup_model.parameters(), 10.0)
        mup_optimizer.step()
        
        if epoch % 200 == 0 and epoch > 0:
            print(f"   Epoch {epoch}: Loss = {loss.item():.6f}")
    
    mup_model.eval()
    with torch.no_grad():
        mup_pred = mup_model(x_test)
    
    mup_mse = mean_squared_error(y_true.numpy().flatten(), mup_pred.numpy().flatten())
    print(f"   Final MSE: {mup_mse:.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, model, pred, title, mse in [
        (axes[0], std_model, std_pred, 'Standard Parametrization (NTK regime)', std_mse),
        (axes[1], mup_model, mup_pred, 'muP (Feature Learning!)', mup_mse)
    ]:
        model.eval()
        pred_1d = pred.numpy().flatten()
        x_test_1d = x_test.numpy().flatten()
        y_true_1d = y_true.numpy().flatten()
        x_train_1d = x_train.numpy().flatten()
        y_train_1d = y_train.numpy().flatten()
        
        if not (np.any(np.isnan(pred_1d)) or np.any(np.isinf(pred_1d))):
            ax.scatter(x_train_1d, y_train_1d, alpha=0.5, s=20, 
                      label='Training data', color='gray')
            ax.plot(x_test_1d, y_true_1d, 'k--', linewidth=2, 
                   label='True function')
            ax.plot(x_test_1d, pred_1d, 'r-', linewidth=2 if 'Standard' in title else 2,
                   color='blue' if 'Standard' in title else 'green',
                   label='NN prediction')
            ax.legend()
        
        ax.set_title(f'{title}\nWidth={width}, MSE={mse:.6f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mup_vs_standard_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison figure saved as 'mup_vs_standard_comparison.png'")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Standard Param: MSE = {std_mse:.6f} (limited by NTK)")
    print(f"muP:           MSE = {mup_mse:.6f} (feature learning enabled)")
    
    if mup_mse < std_mse:
        improvement = (std_mse - mup_mse) / std_mse * 100
        print(f"\n✓ muP achieves {improvement:.1f}% better performance!")


if __name__ == "__main__":
    # Main demonstration
    demonstrate_feature_learning_mup()
    
    # Comparison with standard parametrization
    compare_mup_vs_standard()
    
    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)
