"""
Experiment 3: NTK Limitations in Feature Learning

This script demonstrates that in the infinite-width limit with standard
parameterization, the NTK regime prevents feature learning - the network
can only do kernel regression, not learn new features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from models import WideMLP
from utils import compute_ntk_analytical
from sklearn.metrics import mean_squared_error


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


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 0.01,
    clip_grad_norm: float = 10.0
):
    """
    Train a model on the data.
    
    Args:
        model: Neural network model
        x_train: Training inputs
        y_train: Training targets
        n_epochs: Number of training epochs
        lr: Learning rate
        clip_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
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


def demonstrate_feature_learning_limitations():
    """
    Main demonstration: Show that NTK regime limits feature learning.
    """
    print("=" * 60)
    print("Experiment 3: NTK Limitations in Feature Learning")
    print("=" * 60)
    
    # Generate training data
    n_train = 50
    shift = 1.0  # Task requires learning this shift
    
    x_train, y_train = generate_shifted_data(n_train, shift)
    
    print(f"\nTask: Learn y = sin(x + {shift})")
    print(f"Training samples: {n_train}")
    
    # Test different widths
    widths = [10, 50, 200, 1000, 5000]
    
    final_losses = []
    trained_models = []
    
    for width in widths:
        print(f"\n{'='*40}")
        print(f"Training network with width = {width}")
        print(f"{'='*40}")
        
        # Create model with standard initialization (leads to NTK regime)
        model = WideMLP(1, width, output_dim=1, activation='relu', sigma_w=1.0)
        
        # Scale learning rate inversely with width to prevent instability
        # Standard practice: lr scales as 1/width or sqrt(width)
        base_lr = 0.01
        scaled_lr = base_lr / max(1.0, np.sqrt(width / 100.0))
        
        # Train
        loss_history = train_model(model, x_train, y_train, n_epochs=500, lr=scaled_lr)
        
        final_loss = loss_history[-1]
        final_losses.append(final_loss)
        trained_models.append(model)
        
        if np.isnan(final_loss) or np.isinf(final_loss):
            print(f"Final training loss: NaN/Inf (training unstable)")
        else:
            print(f"Final training loss: {final_loss:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Loss vs width (filter out NaN/Inf for plotting)
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
        ax.plot(valid_widths, valid_losses, 'o-', linewidth=2, markersize=8, label='Valid losses')
    
    if invalid_widths and valid_losses:
        # Mark invalid points at top of plot
        y_max = max(valid_losses) if valid_losses else 1.0
        for w in invalid_widths:
            ax.plot(w, y_max * 1.1, 'rx', markersize=12)
        if len(invalid_widths) > 0:
            ax.plot([], [], 'rx', markersize=12, label='NaN/Inf (unstable)')
    
    ax.set_xscale('log')
    ax.set_xlabel('Network Width')
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Training Loss vs Width')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.01, color='r', linestyle='--', label='Good performance threshold')
    if valid_widths:
        ax.legend()
    
    # Plot 2-6: Predictions for different widths
    x_test = torch.linspace(-3, 3, 200).reshape(-1, 1).float()
    y_true = torch.sin(x_test + shift)
    
    for idx, (width, model) in enumerate(zip(widths, trained_models)):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
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
                ax.plot(x_test_1d, y_pred_1d, 'b-', linewidth=2, 
                       label='NN prediction')
                loss_str = 'NaN' if (np.isnan(final_losses[idx]) or np.isinf(final_losses[idx])) else f'{final_losses[idx]:.4f}'
                ax.legend(fontsize=8)
            
            ax.set_title(f'Width = {width}\nLoss = {loss_str}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_learning_limitations.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved as 'feature_learning_limitations.png'")
    plt.close()
    
    # Demonstrate the key issue: in NTK regime, network can't learn features
    print("\n" + "=" * 60)
    print("Key Observation:")
    print("=" * 60)
    print("\nAs width increases:")
    print("- The network enters the 'lazy training' or 'NTK regime'")
    print("- Parameters don't move much from initialization")
    print("- Network can only do kernel regression with fixed features")
    print("- Cannot learn new features (like the shift in this example)")
    
    print(f"\nLoss comparison:")
    for width, loss in zip(widths, final_losses):
        if np.isnan(loss) or np.isinf(loss):
            print(f"  Width {width:4d}: NaN/Inf (training unstable)")
        else:
            print(f"  Width {width:4d}: {loss:.6f}")
    
    # Check if last loss is valid for comparison
    if not (np.isnan(final_losses[-1]) or np.isinf(final_losses[-1])):
        if final_losses[-1] > final_losses[0] * 1.5:
            print("\n⚠ Notice: Wider networks perform WORSE!")
            print("This is because they're stuck in the NTK regime.")
        else:
            print("\nLoss decreases, but limited by kernel regression capability.")
    else:
        print("\n⚠ Notice: Very wide networks become numerically unstable.")
        print("This demonstrates the need for proper initialization/scaling in wide networks.")
    
    # Compare with a narrow network that CAN learn features
    print("\n" + "=" * 60)
    print("Comparison: Narrow network (can learn features) vs Wide (NTK regime)")
    print("=" * 60)
    
    # Narrow network
    narrow_model = WideMLP(1, width=10, output_dim=1, activation='relu', sigma_w=1.0)
    print("\nTraining narrow network (width=10)...")
    train_model(narrow_model, x_train, y_train, n_epochs=2000, lr=0.05)
    
    # Wide network - use scaled learning rate
    wide_model = WideMLP(1, width=5000, output_dim=1, activation='relu', sigma_w=1.0)
    print("\nTraining wide network (width=5000)...")
    wide_scaled_lr = 0.05 / np.sqrt(5000 / 10.0)  # Scale LR based on width
    train_model(wide_model, x_train, y_train, n_epochs=2000, lr=wide_scaled_lr)
    
    # Evaluate - check for NaN before computing MSE
    with torch.no_grad():
        narrow_pred = narrow_model(x_test)
        wide_pred = wide_model(x_test)
    
    narrow_pred_np = narrow_pred.numpy().flatten()
    wide_pred_np = wide_pred.numpy().flatten()
    y_true_np = y_true.numpy().flatten()
    
    # Compute MSE only if predictions are valid
    if np.any(np.isnan(narrow_pred_np)) or np.any(np.isinf(narrow_pred_np)):
        narrow_mse = np.nan
        print("  Warning: Narrow model predictions contain NaN/Inf")
    else:
        narrow_mse = mean_squared_error(y_true_np, narrow_pred_np)
    
    if np.any(np.isnan(wide_pred_np)) or np.any(np.isinf(wide_pred_np)):
        wide_mse = np.nan
        print("  Warning: Wide model predictions contain NaN/Inf")
    else:
        wide_mse = mean_squared_error(y_true_np, wide_pred_np)
    
    print(f"\nTest MSE:")
    if not np.isnan(narrow_mse):
        print(f"  Narrow (width=10):  {narrow_mse:.6f}")
    else:
        print(f"  Narrow (width=10):  NaN")
    if not np.isnan(wide_mse):
        print(f"  Wide (width=5000):  {wide_mse:.6f}")
    else:
        print(f"  Wide (width=5000):  NaN (unstable)")
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to 1D for matplotlib
    x_train_1d = x_train.numpy().flatten()
    x_test_1d = x_test.numpy().flatten()
    y_train_1d = y_train.numpy().flatten()
    y_true_1d = y_true.numpy().flatten()
    
    for ax, model, title, mse in [(ax1, narrow_model, f'Narrow (width=10)', narrow_mse),
                                  (ax2, wide_model, f'Wide (width=5000)', wide_mse)]:
        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
        
        y_pred_1d = y_pred.numpy().flatten()
        
        # Only plot if predictions are valid
        if np.any(np.isnan(y_pred_1d)) or np.any(np.isinf(y_pred_1d)):
            ax.text(0.5, 0.5, f'Model unstable\nContains NaN/Inf', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            mse_str = 'NaN'
        else:
            ax.scatter(x_train_1d, y_train_1d, alpha=0.5, s=20, 
                      label='Training data', color='gray')
            ax.plot(x_test_1d, y_true_1d, 'k--', linewidth=2, 
                   label='True function')
            ax.plot(x_test_1d, y_pred_1d, 'b-', linewidth=2, 
                   label='NN prediction')
            ax.legend()
            mse_str = f'{mse:.6f}' if not np.isnan(mse) else 'NaN'
        
        ax.set_title(f'{title}\nTest MSE = {mse_str}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('narrow_vs_wide_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison figure saved as 'narrow_vs_wide_comparison.png'")
    plt.close()
    
    print("\n✓ Experiment 3 Complete!")
    print("\nKey Insight: In the infinite-width limit with standard")
    print("parameterization, networks are stuck in the NTK regime and")
    print("cannot learn features - they can only do kernel regression.")


if __name__ == "__main__":
    demonstrate_feature_learning_limitations()

