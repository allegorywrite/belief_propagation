#!/usr/bin/env python3
"""
Factor creation utilities for Gaussian Belief Propagation.
Contains functions for creating different types of factors (prior, linear, range-bearing).
"""

import numpy as np
from typing import Tuple


def create_linear_factor(
    measurement: np.ndarray, 
    measurement_precision: np.ndarray,
    jacobian: np.ndarray, 
    residual_offset: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create factor potential for linear measurement function.
    
    Based on paper equations (11-12):
    η_f = A^T Λ_s (z - b)
    Λ_f = A^T Λ_s A
    
    Args:
        measurement: observed measurement z
        measurement_precision: measurement precision Λ_s
        jacobian: measurement jacobian A  
        residual_offset: offset b (default: zero)
    
    Returns:
        (precision_matrix, information_vector)
    """
    if residual_offset is None:
        residual_offset = np.zeros_like(measurement)
        
    # Compute factor precision: Λ_f = A^T Λ_s A
    factor_precision = jacobian.T @ measurement_precision @ jacobian
    
    # Compute factor information: η_f = A^T Λ_s (z - b)
    factor_information = jacobian.T @ measurement_precision @ (measurement - residual_offset)
    
    return factor_precision, factor_information


def create_prior_factor(
    prior_mean: np.ndarray, 
    prior_precision: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a unary prior factor.
    
    Args:
        prior_mean: prior mean μ
        prior_precision: prior precision Λ
        
    Returns:
        (precision_matrix, information_vector)
    """
    # For prior factor: Λ_f = Λ, η_f = Λ * μ
    return prior_precision, prior_precision @ prior_mean


def create_relative_position_factor(
    true_pos1: np.ndarray,
    true_pos2: np.ndarray,
    relative_precision: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create factor for relative position constraint between two variables.
    
    Measurement: z = x2 - x1 (relative position)
    Jacobian: [-I, I] (2x4 matrix for 2D positions)
    
    Args:
        true_pos1: true position of first variable
        true_pos2: true position of second variable  
        relative_precision: precision matrix for relative measurement
        
    Returns:
        (precision_matrix, information_vector)
    """
    # True relative position measurement
    true_relative = true_pos2 - true_pos1
    
    # Jacobian: [-I, I] (2x4 matrix)
    jacobian = np.array([[-1.0, 0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 1.0]])
    
    return create_linear_factor(true_relative, relative_precision, jacobian)


def create_range_bearing_factor(
    robot1_pos: np.ndarray, 
    robot2_pos: np.ndarray,
    measurement: np.ndarray, 
    measurement_precision: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create factor for range-bearing measurement between two robots.
    
    Args:
        robot1_pos: current estimate of robot 1 position (linearization point)
        robot2_pos: current estimate of robot 2 position 
        measurement: [range, bearing] measurement
        measurement_precision: 2x2 precision matrix for measurement
        
    Returns:
        (4x4 precision matrix, 4x1 information vector)
    """
    # Compute expected measurement h(x1, x2)
    diff = robot2_pos - robot1_pos
    expected_range = np.linalg.norm(diff)
    expected_bearing = np.arctan2(diff[1], diff[0])
    expected_measurement = np.array([expected_range, expected_bearing])
    
    # Compute Jacobian of h(x1, x2) = [||x2-x1||, atan2(y2-y1, x2-x1)]
    if expected_range > 1e-6:  # Avoid division by zero
        # ∂range/∂x1 = -(x2-x1)/||x2-x1||, ∂range/∂x2 = (x2-x1)/||x2-x1||
        range_grad = diff / expected_range
        
        # ∂bearing/∂x1, ∂bearing/∂x2
        dist_sq = expected_range ** 2
        bearing_grad_x1 = np.array([diff[1], -diff[0]]) / dist_sq
        bearing_grad_x2 = -bearing_grad_x1
        
        # Jacobian matrix: 2x4 [∂h/∂x1, ∂h/∂x2]
        jacobian = np.array([
            [-range_grad[0], -range_grad[1], range_grad[0], range_grad[1]],
            [bearing_grad_x1[0], bearing_grad_x1[1], bearing_grad_x2[0], bearing_grad_x2[1]]
        ])
    else:
        # Fallback for very small distances
        jacobian = np.array([
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0]
        ])
        
    # Residual offset b = h(x0) - J*x0 where x0 = [x1; x2]
    state_vector = np.concatenate([robot1_pos, robot2_pos])
    residual_offset = expected_measurement - jacobian @ state_vector
    
    return create_linear_factor(measurement, measurement_precision, jacobian, residual_offset)


def create_smoothness_factor(
    current_pos1: np.ndarray,
    current_pos2: np.ndarray,
    smoothness_precision: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create smoothness factor to maintain current estimated relative positions.
    
    Args:
        current_pos1: current estimated position of first variable
        current_pos2: current estimated position of second variable
        smoothness_precision: precision matrix for smoothness constraint
        
    Returns:
        (precision_matrix, information_vector)
    """
    # Current estimated relative position
    estimated_relative = current_pos2 - current_pos1
    
    # Jacobian: [-I, I] (2x4 matrix)
    jacobian = np.array([[-1.0, 0.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 1.0]])
    
    return create_linear_factor(estimated_relative, smoothness_precision, jacobian)


def compute_range_bearing(pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
    """
    Compute range-bearing measurement h(x1, x2) = [r, b]
    
    Args:
        pos1: position of first robot
        pos2: position of second robot
        
    Returns:
        [range, bearing] measurement
    """
    diff = pos2 - pos1
    range_val = np.linalg.norm(diff)
    bearing = np.arctan2(diff[1], diff[0])
    return np.array([range_val, bearing])