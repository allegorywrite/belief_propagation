#!/usr/bin/env python3
"""
Gaussian message data structures for Belief Propagation
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GaussianMessage:
    """Gaussian message representation with mean and precision matrix."""
    mean: np.ndarray
    precision: np.ndarray
    
    @property
    def covariance(self) -> np.ndarray:
        return np.linalg.inv(self.precision)