#!/usr/bin/env python3
"""
Utility modules for Gaussian Belief Propagation
"""

from .linear_algebra_utils import (
    schur_complement_marginalization,
    regularize_precision_matrix,
    safe_precision_to_mean,
    combine_gaussian_messages,
    block_matrix_inverse
)
from .factor_utils import (
    create_linear_factor,
    create_prior_factor,
    create_relative_position_factor,
    create_range_bearing_factor,
    create_smoothness_factor,
    compute_range_bearing
)
from .graph_utils import GraphFactory, PositionManager
from .bp_problem_setup import BPProblemSetup

__all__ = [
    'schur_complement_marginalization',
    'regularize_precision_matrix',
    'safe_precision_to_mean',
    'combine_gaussian_messages',
    'block_matrix_inverse',
    'create_linear_factor',
    'create_prior_factor',
    'create_relative_position_factor',
    'create_range_bearing_factor',
    'create_smoothness_factor',
    'compute_range_bearing',
    'GraphFactory',
    'PositionManager',
    'BPProblemSetup'
]