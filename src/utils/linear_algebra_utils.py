#!/usr/bin/env python3
"""
Linear algebra utilities for Gaussian Belief Propagation.
Contains Schur complement operations and matrix manipulation functions.
"""

import numpy as np
from typing import Tuple, Optional


def schur_complement_marginalization(
    precision_matrix: np.ndarray,
    information_vector: np.ndarray,
    var_dim: int,
    keep_first: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Marginalize out variables using Schur complement.
    
    For a 2-variable system with precision matrix:
    P = [[P11, P12],
         [P21, P22]]
    
    And information vector:
    eta = [eta1, eta2]
    
    Args:
        precision_matrix: Block precision matrix (4x4 for 2D variables)
        information_vector: Information vector (4x1 for 2D variables)
        var_dim: Dimension of each variable (typically 2)
        keep_first: If True, keep first variable and marginalize second
                   If False, keep second variable and marginalize first
    
    Returns:
        (marginal_precision, marginal_information): The marginalized result
    """
    if precision_matrix.shape[0] != 2 * var_dim:
        raise ValueError(f"Expected {2*var_dim}x{2*var_dim} precision matrix")
    
    # Extract blocks
    P11 = precision_matrix[0:var_dim, 0:var_dim]
    P12 = precision_matrix[0:var_dim, var_dim:2*var_dim] 
    P21 = precision_matrix[var_dim:2*var_dim, 0:var_dim]
    P22 = precision_matrix[var_dim:2*var_dim, var_dim:2*var_dim]
    
    eta1 = information_vector[0:var_dim]
    eta2 = information_vector[var_dim:2*var_dim]
    
    if keep_first:
        # Keep first variable, marginalize second
        # Marginal precision: P11 - P12 * P22^(-1) * P21
        # Marginal information: eta1 - P12 * P22^(-1) * eta2
        try:
            P22_inv = np.linalg.inv(P22)
            marginal_precision = P11 - P12 @ P22_inv @ P21
            marginal_information = eta1 - P12 @ P22_inv @ eta2
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            P22_pinv = np.linalg.pinv(P22)
            marginal_precision = P11 - P12 @ P22_pinv @ P21
            marginal_information = eta1 - P12 @ P22_pinv @ eta2
    else:
        # Keep second variable, marginalize first
        # Marginal precision: P22 - P21 * P11^(-1) * P12
        # Marginal information: eta2 - P21 * P11^(-1) * eta1
        try:
            P11_inv = np.linalg.inv(P11)
            marginal_precision = P22 - P21 @ P11_inv @ P12
            marginal_information = eta2 - P21 @ P11_inv @ eta1
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            P11_pinv = np.linalg.pinv(P11)
            marginal_precision = P22 - P21 @ P11_pinv @ P12
            marginal_information = eta2 - P21 @ P11_pinv @ eta1
    
    return marginal_precision, marginal_information


def regularize_precision_matrix(precision: np.ndarray, reg_factor: float = 1e-8) -> np.ndarray:
    """
    Add regularization to precision matrix to prevent singularity.
    
    Args:
        precision: Input precision matrix
        reg_factor: Regularization factor (added to diagonal)
    
    Returns:
        Regularized precision matrix
    """
    return precision + reg_factor * np.eye(precision.shape[0])


def safe_precision_to_mean(precision: np.ndarray, information: np.ndarray) -> np.ndarray:
    """
    Safely convert precision/information form to mean.
    
    Args:
        precision: Precision matrix
        information: Information vector
    
    Returns:
        Mean vector (precision^(-1) * information)
    """
    try:
        return np.linalg.solve(precision, information)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        try:
            return np.linalg.pinv(precision) @ information
        except:
            # Ultimate fallback: return zero mean
            return np.zeros(information.shape[0])


def combine_gaussian_messages(
    precisions: list[np.ndarray], 
    means: list[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine multiple Gaussian messages by summing precisions and information.
    
    Args:
        precisions: List of precision matrices
        means: List of mean vectors
    
    Returns:
        (combined_precision, combined_mean): The combined Gaussian
    """
    if not precisions:
        raise ValueError("Empty precision list")
    
    # Sum precisions and information vectors
    combined_precision = np.zeros_like(precisions[0])
    combined_information = np.zeros_like(means[0])
    
    for precision, mean in zip(precisions, means):
        combined_precision += precision
        combined_information += precision @ mean
    
    # Add regularization
    combined_precision = regularize_precision_matrix(combined_precision)
    
    # Convert back to mean form
    combined_mean = safe_precision_to_mean(combined_precision, combined_information)
    
    return combined_precision, combined_mean


def block_matrix_inverse(block_matrix: np.ndarray, block_size: int) -> np.ndarray:
    """
    Compute inverse of block matrix using block inversion formulas.
    
    For 2x2 block matrix:
    [[A, B],     -->    [[A_inv + A_inv*B*S_inv*C*A_inv, -A_inv*B*S_inv],
     [C, D]]             [-S_inv*C*A_inv,                  S_inv]]
    
    where S = D - C*A_inv*B (Schur complement)
    
    Args:
        block_matrix: Input matrix to invert
        block_size: Size of each block
    
    Returns:
        Inverse matrix
    """
    if block_matrix.shape[0] != 2 * block_size:
        return np.linalg.pinv(block_matrix)
    
    try:
        A = block_matrix[0:block_size, 0:block_size]
        B = block_matrix[0:block_size, block_size:2*block_size]
        C = block_matrix[block_size:2*block_size, 0:block_size]
        D = block_matrix[block_size:2*block_size, block_size:2*block_size]
        
        A_inv = np.linalg.inv(A)
        S = D - C @ A_inv @ B  # Schur complement
        S_inv = np.linalg.inv(S)
        
        # Build inverse using block formulas
        inv_11 = A_inv + A_inv @ B @ S_inv @ C @ A_inv
        inv_12 = -A_inv @ B @ S_inv
        inv_21 = -S_inv @ C @ A_inv
        inv_22 = S_inv
        
        inverse = np.block([[inv_11, inv_12],
                           [inv_21, inv_22]])
        
        return inverse
        
    except np.linalg.LinAlgError:
        # Fallback to standard pseudo-inverse
        return np.linalg.pinv(block_matrix)