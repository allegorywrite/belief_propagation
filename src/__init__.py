#!/usr/bin/env python3
"""
Gaussian Belief Propagation package

A modular implementation of Gaussian Belief Propagation with visualization
"""

from .gaussian_message import GaussianMessage
from .gaussian_bp import GaussianBP
from .graph_utils import GraphFactory, PositionManager
from .animation import DynamicEdgeBPAnimation

__all__ = [
    'GaussianMessage',
    'GaussianBP', 
    'GraphFactory',
    'PositionManager',
    'DynamicEdgeBPAnimation'
]