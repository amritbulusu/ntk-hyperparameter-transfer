"""
Neural Tangent Kernel experiments package.
"""

from .models import MLP, WideMLP
try:
    from .models_mup import WideMLP_MuP, MuLinear, MuReadout
    _mup_available = True
except ImportError:
    _mup_available = False
    # muP models not available
from .utils import (
    compute_ntk_kernel,
    compute_ntk_analytical,
    init_weights_gaussian,
    plot_kernel_matrix,
    sample_gp_from_kernel
)

__all__ = [
    'MLP',
    'WideMLP',
    'compute_ntk_kernel',
    'compute_ntk_analytical',
    'init_weights_gaussian',
    'plot_kernel_matrix',
    'sample_gp_from_kernel',
]

if _mup_available:
    __all__.extend(['WideMLP_MuP', 'MuLinear', 'MuReadout'])

