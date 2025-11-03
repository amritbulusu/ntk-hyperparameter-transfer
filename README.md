# Neural Tangent Kernel Experiments

This repository contains experiments demonstrating key concepts in Neural Tangent Kernel (NTK) theory:

1. **Neural Networks as Gaussian Processes**: Shows how infinite-width neural networks behave as Gaussian Processes
2. **Neural Tangent Kernel Properties**: Explores the NTK and its mathematical properties
3. **NTK Limitations in Feature Learning**: Demonstrates how the NTK fails to learn features in the infinite-width limit
4. **Feature Learning with muP**: Shows how Maximal Update Parametrization (muP) enables feature learning at infinite width

## Setup

```bash
pip install -r requirements.txt
```

## Structure

- `ntk/`: Main experiments directory
  - `gaussian_process_demo.py`: Demonstrates neural networks as Gaussian Processes
  - `ntk_properties.py`: Explores NTK properties and computation
  - `feature_learning_limitations.py`: Shows NTK limitations in feature learning
  - `feature_learning_mup.py`: Demonstrates feature learning with muP at infinite width ⭐
  - `utils.py`: Utility functions for kernel computations and visualization
  - `models.py`: Neural network architectures
  - `models_mup.py`: muP-based architectures for feature learning

## Experiments

Each experiment script can be run independently and includes visualization of results.

### Running Experiments

1. **Neural Networks as Gaussian Processes**:
   ```bash
   python ntk/gaussian_process_demo.py
   ```
   This demonstrates how infinite-width neural networks converge to Gaussian Processes.

2. **NTK Properties**:
   ```bash
   python ntk/ntk_properties.py
   ```
   Explores the Neural Tangent Kernel and its mathematical properties.

3. **Feature Learning Limitations**:
   ```bash
   python ntk/feature_learning_limitations.py
   ```
   Shows how the NTK regime prevents feature learning in wide networks.

4. **Feature Learning with muP** ⭐:
   ```bash
   python ntk/feature_learning_mup.py
   ```
   Demonstrates how Maximal Update Parametrization (muP) enables feature learning even as width → ∞. This is the key solution to NTK limitations!

## Theory

The Neural Tangent Kernel (NTK) is a key theoretical tool that connects neural networks to kernel methods. Key concepts:

- **Infinite-width limit**: As network width → ∞, neural networks behave as Gaussian Processes
- **NTK definition**: K(x, x') = ∇_θ f(x) · ∇_θ f(x'), the inner product of gradients
- **Training dynamics**: df/dt = -η K (f - y), where the NTK determines training speed
- **Feature learning limitation**: In the NTK regime, networks can only do kernel regression with fixed features
- **muP Solution**: Maximal Update Parametrization enables feature learning by stabilizing hyperparameters across widths

## Key Results

1. **NTK Regime Problem**: With standard parametrization, wide networks get stuck in the NTK regime and cannot learn features
2. **muP Solution**: Maximal Update Parametrization uniquely enables feature learning at infinite width while maintaining stable hyperparameters
3. **Practical Impact**: With muP, you can tune hyperparameters on small models and transfer them to arbitrarily large models

