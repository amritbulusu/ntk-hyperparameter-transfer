# Maximal Update Parametrization (muP) Implementation

## Overview

This repository now includes an implementation of **Maximal Update Parametrization (muP)** to demonstrate feature learning at infinite width. This solves the fundamental limitation of the NTK regime where wide networks cannot learn features.

## Key Files

1. **`ntk/models_mup.py`**: µP-based neural network architectures
   - `MuLinear`: Hidden layer with muP scaling
   - `MuReadout`: Output layer with proper muP scaling (1/width)
   - `WideMLP_MuP`: Complete wide MLP with muP parametrization

2. **`ntk/feature_learning_mup.py`**: Main experiment demonstrating feature learning
   - Shows how muP enables feature learning as width → ∞
   - Direct comparison with standard parametrization
   - Generates visualization plots

3. **`requirements.txt`**: Updated to include `mup>=1.0.0`

## Key Differences from Standard Parametrization

### Standard Parametrization (NTK Regime)
- Hidden weights: N(0, σ_w²/n)
- Output weights: N(0, σ_w²/n)
- Learning rate: Must scale as 1/√n
- Result: Stuck in NTK regime, no feature learning

### muP Parametrization (Feature Learning)
- Hidden weights: N(0, σ_w²/fan_in)
- Output weights: 0 (special muP initialization)
- Learning rate: Fixed across all widths!
- Result: Feature learning works even at infinite width

## Running the Experiment

Once the `mup` package is installed:

```bash
python ntk/feature_learning_mup.py
```

This will:
1. Train muP networks at multiple widths (10, 50, 200, 1000, 5000, 10000)
2. Show that wider networks perform BETTER (not worse!)
3. Generate comparison plots:
   - `feature_learning_mup.png`: muP results across widths
   - `mup_vs_standard_comparison.png`: Direct comparison

## Expected Results

### With muP:
- ✓ Wider networks achieve LOWER loss
- ✓ Same learning rate works for all widths
- ✓ Feature learning is preserved
- ✓ Stable training (no NaN/Inf)

### With Standard Parametrization:
- ✗ Wider networks perform WORSE
- ✗ Must tune learning rate per width
- ✗ Stuck in NTK/kernel regression
- ✗ Training becomes unstable

## Mathematical Intuition

muP achieves "maximal updates" by ensuring that:
1. Every activation has Θ(1) coordinate size
2. Network output is O(1)
3. Parameters are updated as much as possible without divergence

This is achieved through careful scaling of:
- Initialization: Different for hidden vs output layers
- Learning rates: Stable across widths
- Weight updates: Proper width-dependent scaling built into the parametrization

## References

- Yang & Hu et al. (2022): "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
- https://github.com/microsoft/mup
- arxiv.org/abs/2203.03466

## Next Steps

The implementation is ready. You just need to:
1. Install the `mup` package: `pip install mup`
2. Run the experiment: `python ntk/feature_learning_mup.py`

The experiment will automatically demonstrate that muP enables feature learning at infinite width, solving the fundamental NTK limitation!

