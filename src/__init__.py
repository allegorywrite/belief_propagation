#!/usr/bin/env python3
"""
Gaussian Belief Propagation package

A modular implementation of Gaussian Belief Propagation with visualization
"""

from .gaussian_message import GaussianMessage
from .gaussian_bp import GaussianBP
from .particle_node import ParticleNode, SVGDOptimizer
from .utils import GraphFactory, PositionManager, BPProblemSetup
from .visualization import DynamicEdgeBPAnimation
from .utils import (
    schur_complement_marginalization,
    regularize_precision_matrix,
    safe_precision_to_mean,
    create_linear_factor,
    create_prior_factor, 
    create_relative_position_factor
)

__all__ = [
    'GaussianMessage',
    'GaussianBP',
    'ParticleNode', 
    'SVGDOptimizer',
    'GraphFactory',
    'PositionManager',
    'DynamicEdgeBPAnimation',
    'BPProblemSetup',
    'schur_complement_marginalization',
    'regularize_precision_matrix',
    'safe_precision_to_mean',
    'create_linear_factor',
    'create_prior_factor',
    'create_relative_position_factor'
]