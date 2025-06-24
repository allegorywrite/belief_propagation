#!/usr/bin/env python3
"""
Particle Node implementation for Belief Propagation with SVGD
"""

import numpy as np
from typing import Callable, Optional
from .gaussian_message import GaussianMessage


class ParticleNode:
    """A node that maintains a set of particles instead of a Gaussian distribution."""
    
    def __init__(self, num_particles: int, dim: int, 
                 initial_particles: Optional[np.ndarray] = None):
        """
        Initialize particle node.
        
        Args:
            num_particles: Number of particles to maintain
            dim: Dimensionality of each particle
            initial_particles: Optional initial particle positions, shape (num_particles, dim)
        """
        self.num_particles = num_particles
        self.dim = dim
        
        if initial_particles is not None:
            assert initial_particles.shape == (num_particles, dim)
            self.particles = initial_particles.copy()
        else:
            # Initialize particles randomly
            self.particles = np.random.randn(num_particles, dim)
        
        # Equal weights for all particles initially
        self.weights = np.ones(num_particles) / num_particles
    
    def update_particles_svgd(self, log_prob_grad_fn: Callable[[np.ndarray], np.ndarray], 
                             step_size: float, kernel_bandwidth: float = 1.0):
        """
        Update particles using Stein Variational Gradient Descent.
        
        Args:
            log_prob_grad_fn: Function that takes particle array and returns gradients
            step_size: SVGD step size
            kernel_bandwidth: RBF kernel bandwidth
        """
        # print(f"[DEBUG] update_particles_svgd called - particles shape: {self.particles.shape}")
        
        # Compute log probability gradients for all particles
        log_prob_grads = log_prob_grad_fn(self.particles)  # Shape: (num_particles, dim)
        
        # Compute SVGD update
        svgd_update = self._compute_svgd_update(log_prob_grads, kernel_bandwidth)
        
        # Update particles
        self.particles += step_size * svgd_update
        
        # Update particle weights using kernel-based error updates
        self._update_particle_weights(log_prob_grads, kernel_bandwidth)
    
    def _compute_svgd_update(self, log_prob_grads: np.ndarray, 
                            kernel_bandwidth: float) -> np.ndarray:
        """
        Compute SVGD update direction.
        
        Args:
            log_prob_grads: Gradients of log probability, shape (num_particles, dim)
            kernel_bandwidth: RBF kernel bandwidth
            
        Returns:
            SVGD update direction, shape (num_particles, dim)
        """
        # print(f"[DEBUG] _compute_svgd_update called with {log_prob_grads.shape[0]} particles")
        # print(f"[DEBUG] log_prob_grads norm: {np.linalg.norm(log_prob_grads):.6f}")
        # print(f"[DEBUG] kernel_bandwidth: {kernel_bandwidth:.6f}")
        
        n_particles = self.particles.shape[0]
        update = np.zeros_like(self.particles)
        
        for i in range(n_particles):
            phi_i = np.zeros(self.dim)
            
            for j in range(n_particles):
                # Compute RBF kernel and its gradient
                diff = self.particles[j] - self.particles[i]
                kernel_val = self._rbf_kernel(diff, kernel_bandwidth)
                kernel_grad = self._rbf_kernel_gradient(diff, kernel_bandwidth)
                
                # SVGD update rule
                phi_i += kernel_val * log_prob_grads[i] + 1.0 * kernel_grad
                # phi_i += log_prob_grads[i]
            
            update[i] = phi_i / n_particles
            # print(update[i])
        
        return update
    
    def _update_particle_weights(self, log_prob_grads: np.ndarray, kernel_bandwidth: float):
        """
        Update particle weights using kernel-based error updates with normalization.
        
        Args:
            log_prob_grads: Gradients of log probability, shape (num_particles, dim)
            kernel_bandwidth: RBF kernel bandwidth
        """
        n_particles = self.particles.shape[0]
        new_weights = np.zeros(n_particles)
        
        for i in range(n_particles):
            # Compute weight update based on kernel interactions and gradient errors
            weight_update = 0.0
            
            for j in range(n_particles):
                # Compute kernel value between particles i and j
                diff = self.particles[j] - self.particles[i]
                kernel_val = self._rbf_kernel(diff, kernel_bandwidth)
                
                # Error-based weight update using gradient norms
                grad_error = np.linalg.norm(log_prob_grads[i])
                
                # Weight update proportional to kernel and inversely proportional to error
                weight_update += kernel_val / (1.0 + grad_error)
            
            new_weights[i] = weight_update / n_particles
        
        # Normalize weights to sum to 1
        if np.sum(new_weights) > 0:
            self.weights = new_weights / np.sum(new_weights)
        else:
            # Fallback to uniform weights if all weights are zero
            self.weights = np.ones(n_particles) / n_particles
    
    def _rbf_kernel(self, diff: np.ndarray, bandwidth: float) -> float:
        """Compute RBF kernel value."""
        return np.exp(-np.sum(diff**2) / (2 * bandwidth**2))
    
    def _rbf_kernel_gradient(self, diff: np.ndarray, bandwidth: float) -> np.ndarray:
        """Compute gradient of RBF kernel."""
        kernel_val = self._rbf_kernel(diff, bandwidth)
        return -kernel_val * diff / (bandwidth**2)
    
    def compute_gaussian_approximation(self) -> GaussianMessage:
        """
        Compute Gaussian approximation of particle distribution using maximum likelihood estimation.
        
        Returns:
            GaussianMessage representing the approximated distribution
        """
        # Use maximum likelihood estimation: particle with highest weight
        max_weight_idx = np.argmax(self.weights)
        mean = self.particles[max_weight_idx]
        
        # Compute weighted covariance
        diff = self.particles - mean[np.newaxis, :]
        cov = np.average(diff[:, :, np.newaxis] * diff[:, np.newaxis, :], 
                        weights=self.weights, axis=0)
        
        # Add small regularization to ensure positive definiteness
        cov += 1e-6 * np.eye(self.dim)
        
        # Convert to precision matrix
        precision = np.linalg.inv(cov)
        
        return GaussianMessage(mean, precision)
    
    def resample_particles(self, importance_weights: np.ndarray):
        """
        Resample particles based on importance weights.
        
        Args:
            importance_weights: Weights for resampling, shape (num_particles,)
        """
        # Normalize weights
        weights = importance_weights / np.sum(importance_weights)
        
        # Systematic resampling
        indices = np.random.choice(self.num_particles, size=self.num_particles, 
                                 p=weights, replace=True)
        
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_particle_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.particles.copy()
    
    def get_particle_weights(self) -> np.ndarray:
        """Get current particle weights."""
        return self.weights.copy()
    
    def set_particles(self, particles: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Set particle positions and optionally weights.
        
        Args:
            particles: New particle positions, shape (num_particles, dim)
            weights: Optional new weights, shape (num_particles,)
        """
        assert particles.shape == (self.num_particles, self.dim)
        self.particles = particles.copy()
        
        if weights is not None:
            assert weights.shape == (self.num_particles,)
            assert np.abs(np.sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
            self.weights = weights.copy()
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles


class SVGDOptimizer:
    """Stein Variational Gradient Descent optimizer."""
    
    def __init__(self, kernel_bandwidth: float = 5.0, adaptive_bandwidth: bool = False):
        """
        Initialize SVGD optimizer.
        
        Args:
            kernel_bandwidth: Initial RBF kernel bandwidth
            adaptive_bandwidth: Whether to use adaptive bandwidth selection
        """
        self.kernel_bandwidth = kernel_bandwidth
        self.adaptive_bandwidth = adaptive_bandwidth
    
    def compute_adaptive_bandwidth(self, particles: np.ndarray) -> float:
        """
        Compute adaptive bandwidth using median heuristic.
        
        Args:
            particles: Current particle positions, shape (num_particles, dim)
            
        Returns:
            Adaptive bandwidth value
        """
        n_particles = particles.shape[0]
        if n_particles <= 1:
            return self.kernel_bandwidth
        
        # Compute pairwise distances
        distances = []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                dist = np.linalg.norm(particles[i] - particles[j])
                distances.append(dist)
        
        if not distances:
            return self.kernel_bandwidth
        
        # Use median distance as bandwidth
        median_dist = np.median(distances)
        return max(median_dist / np.log(n_particles), 1e-6)
    
    def step(self, particle_node: ParticleNode, 
             log_prob_grad_fn: Callable[[np.ndarray], np.ndarray], 
             step_size: float):
        """
        Perform one SVGD step.
        
        Args:
            particle_node: ParticleNode to update
            log_prob_grad_fn: Function that computes log probability gradients
            step_size: Step size for the update
        """
        # Compute adaptive bandwidth if requested
        if self.adaptive_bandwidth:
            bandwidth = self.compute_adaptive_bandwidth(particle_node.particles)
        else:
            bandwidth = self.kernel_bandwidth
        
        # Update particles using SVGD
        particle_node.update_particles_svgd(log_prob_grad_fn, step_size, bandwidth)