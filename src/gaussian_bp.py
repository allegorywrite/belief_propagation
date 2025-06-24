#!/usr/bin/env python3
"""
Gaussian Belief Propagation algorithm implementation with Particle Node support
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, Callable
from .gaussian_message import GaussianMessage
from .particle_node import ParticleNode, SVGDOptimizer
from .utils.linear_algebra_utils import (
    schur_complement_marginalization,
    regularize_precision_matrix,
    safe_precision_to_mean,
    combine_gaussian_messages
)
from .utils.factor_utils import (
    create_linear_factor,
    create_prior_factor,
    create_range_bearing_factor,
    compute_range_bearing
)


class GaussianBP:
    """Gaussian Belief Propagation implementation with Particle Node support."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.messages = {}
        self.beliefs = {}
        self.factor_potentials = {}
        
        # Particle node support
        self.particle_nodes: Dict[str, ParticleNode] = {}
        self.svgd_optimizer = SVGDOptimizer()
        self.svgd_step_size = 0.01
        self.svgd_num_steps = 10
        
    def set_factor_potential(self, factor_node: str, precision: np.ndarray, 
                           information_vector: np.ndarray):
        """Set the potential function for a factor node."""
        self.factor_potentials[factor_node] = {
            'precision': precision,
            'information': information_vector
        }
    
    def initialize_messages(self, dim: int = 2):
        """Initialize all messages to uniform (zero precision)."""
        self.variable_dim = dim  # Store dimension for consistency
        for edge in self.graph.edges():
            node1, node2 = edge
            small_precision = 1e-6 * np.eye(dim)
            zero_mean = np.zeros(dim)
            
            self.messages[(node1, node2)] = GaussianMessage(zero_mean, small_precision)
            self.messages[(node2, node1)] = GaussianMessage(zero_mean, small_precision)
    
    def variable_to_factor_message(self, var_node: str, factor_node: str) -> GaussianMessage:
        """Compute variable-to-factor message according to GBP paper.
        
        The message from variable to factor is the product of all incoming 
        messages except from the target factor.
        """
        neighbors = list(self.graph.neighbors(var_node))
        
        # If only connected to one factor, send uniform message
        if len(neighbors) <= 1:
            dim = getattr(self, 'variable_dim', 2)
            return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
        # Product of all incoming messages except from target factor
        dim = getattr(self, 'variable_dim', 2)
        precision_sum = np.zeros((dim, dim))
        information_sum = np.zeros(dim)
        
        for neighbor in neighbors:
            if neighbor != factor_node:
                msg = self.messages.get((neighbor, var_node))
                if msg is not None:
                    precision_sum += msg.precision
                    information_sum += msg.precision @ msg.mean
        
        # Regularize and convert to mean form
        precision_sum = regularize_precision_matrix(precision_sum)
        mean = safe_precision_to_mean(precision_sum, information_sum)
        
        return GaussianMessage(mean, precision_sum)
    
    def factor_to_variable_message(self, factor_node: str, var_node: str) -> GaussianMessage:
        """Compute factor-to-variable message using proper marginalization.
        
        According to GBP paper equation (14): 
        m_{f->v_i}(v_i) = sum_{V_f \\ v_i} f(V_f) prod_{v_j in V_f \\ v_i} m_{v_j->f}(v_j)
        """
        if factor_node not in self.factor_potentials:
            dim = getattr(self, 'variable_dim', 2)
            return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
        factor_potential = self.factor_potentials[factor_node]
        factor_precision = factor_potential['precision'].copy()
        factor_info = factor_potential['information'].copy()
        
        # Get all variables connected to this factor except target variable
        connected_vars = [n for n in self.graph.neighbors(factor_node) 
                         if not n.startswith('f_') and not n.startswith('obs_')]
        other_vars = [v for v in connected_vars if v != var_node]
        
        # Case 1: Unary factor (only connected to target variable)
        if not other_vars:
            mean = safe_precision_to_mean(factor_precision, factor_info)
            return GaussianMessage(mean, factor_precision)
        
        # Case 2: Binary factor (connected to target + one other variable)
        if len(other_vars) == 1:
            other_var = other_vars[0]
            msg_from_other = self.messages.get((other_var, factor_node))
            
            # Determine variable ordering
            all_vars = sorted(connected_vars)
            var_idx = all_vars.index(var_node)
            other_idx = all_vars.index(other_var)
            
            # Use Schur complement marginalization for binary factors
            var_dim = getattr(self, 'variable_dim', 2)
            expected_size = var_dim * 2  # Two variables
            if factor_precision.shape[0] == expected_size:
                # Augment factor potential with incoming message
                if msg_from_other is not None:
                    if var_idx == 0:  # Target variable is first, add message to second block
                        factor_precision[var_dim:2*var_dim, var_dim:2*var_dim] += msg_from_other.precision
                        factor_info[var_dim:2*var_dim] += msg_from_other.precision @ msg_from_other.mean
                    else:  # Target variable is second, add message to first block
                        factor_precision[0:var_dim, 0:var_dim] += msg_from_other.precision
                        factor_info[0:var_dim] += msg_from_other.precision @ msg_from_other.mean
                
                # Marginalize using Schur complement
                try:
                    keep_first = (var_idx == 0)
                    marginal_precision, marginal_info = schur_complement_marginalization(
                        factor_precision, factor_info, var_dim, keep_first
                    )
                    
                    # Regularize and convert to mean
                    marginal_precision = regularize_precision_matrix(marginal_precision)
                    marginal_mean = safe_precision_to_mean(marginal_precision, marginal_info)
                    
                    return GaussianMessage(marginal_mean, marginal_precision)
                    
                except Exception:
                    # Fallback for numerical issues
                    dim = getattr(self, 'variable_dim', 2)
                    return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
            
            # Fallback for other matrix sizes
            var_dim = getattr(self, 'variable_dim', 2)
            mean = safe_precision_to_mean(factor_precision, factor_info)
            return GaussianMessage(mean[:var_dim], factor_precision[:var_dim, :var_dim])
        
        # Case 3: Higher-order factors (not implemented yet)
        dim = getattr(self, 'variable_dim', 2)
        return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
    
    def update_messages(self) -> float:
        """Perform one iteration of message passing.
        
        Updates all messages in a single pass to avoid ordering issues.
        """
        new_messages = {}
        max_change = 0.0
        
        # Update all variable-to-factor messages first
        for edge in self.graph.edges():
            node1, node2 = edge

            # Identify factor and variable nodes
            if node1.startswith('f_') or node1.startswith('obs_'):
                factor_node, var_node = node1, node2
            else:
                factor_node, var_node = node2, node1
            
            # Update variable-to-factor message
            old_msg = self.messages.get((var_node, factor_node))
            new_msg = self.variable_to_factor_message(var_node, factor_node)
            new_messages[(var_node, factor_node)] = new_msg
            
            if old_msg is not None:
                change = np.linalg.norm(new_msg.mean - old_msg.mean)
                max_change = max(max_change, change)
        
        # Update messages dictionary for factor-to-variable computation
        self.messages.update(new_messages)
        
        # Now update all factor-to-variable messages
        for edge in self.graph.edges():
            node1, node2 = edge
            
            if node1.startswith('f_') or node1.startswith('obs_'):
                factor_node, var_node = node1, node2
            else:
                factor_node, var_node = node2, node1
            
            # Update factor-to-variable message
            old_msg = self.messages.get((factor_node, var_node))
            new_msg = self.factor_to_variable_message(factor_node, var_node)
            new_messages[(factor_node, var_node)] = new_msg
            
            if old_msg is not None:
                change = np.linalg.norm(new_msg.mean - old_msg.mean)
                max_change = max(max_change, change)
        
        # Final update
        self.messages.update(new_messages)
        return max_change
    
    def compute_beliefs(self):
        """Compute marginal beliefs for all variable nodes."""
        for node in self.graph.nodes():
            if not node.startswith('f_') and not node.startswith('obs_'):
                neighbors = list(self.graph.neighbors(node))
                
                if not neighbors:
                    # Isolated node - set to uniform prior
                    dim = getattr(self, 'variable_dim', 2)
                    self.beliefs[node] = GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
                    continue
                
                precision_sum = None
                information_sum = None
                
                for neighbor in neighbors:
                    msg = self.messages.get((neighbor, node))
                    if msg is not None:
                        if precision_sum is None:
                            precision_sum = msg.precision.copy()
                            information_sum = msg.precision @ msg.mean
                        else:
                            precision_sum += msg.precision
                            information_sum += msg.precision @ msg.mean
                
                if precision_sum is not None:
                    precision_sum = regularize_precision_matrix(precision_sum)
                    mean = safe_precision_to_mean(precision_sum, information_sum)
                    self.beliefs[node] = GaussianMessage(mean, precision_sum)
                else:
                    # No messages received - set to weak prior
                    dim = getattr(self, 'variable_dim', 2)
                    weak_precision = 1e-4 * np.eye(dim)
                    self.beliefs[node] = GaussianMessage(np.zeros(dim), weak_precision)
    
    def add_particle_node(self, node_id: str, num_particles: int, dim: int, 
                         initial_particles: Optional[np.ndarray] = None):
        """
        Add a particle node to the graph.
        
        Args:
            node_id: Node identifier
            num_particles: Number of particles
            dim: Dimensionality of particles
            initial_particles: Optional initial particle positions
        """
        self.particle_nodes[node_id] = ParticleNode(num_particles, dim, initial_particles)
    
    def is_particle_node(self, node_id: str) -> bool:
        """Check if a node is a particle node."""
        return node_id in self.particle_nodes
    
    def particle_to_gaussian_message(self, particle_node_id: str, target_node: str) -> GaussianMessage:
        """
        Compute message from particle node to Gaussian node.
        
        This method approximates the particle distribution using moment matching
        and sends the resulting Gaussian message.
        """
        if particle_node_id not in self.particle_nodes:
            dim = getattr(self, 'variable_dim', 2)
            return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
        particle_node = self.particle_nodes[particle_node_id]
        
        # Find the factor connecting these nodes
        connecting_factors = []
        for neighbor in self.graph.neighbors(particle_node_id):
            if neighbor.startswith('f_') and target_node in self.graph.neighbors(neighbor):
                connecting_factors.append(neighbor)
        
        if not connecting_factors:
            # Direct connection or no connection
            return particle_node.compute_gaussian_approximation()
        
        # Use the first connecting factor for now
        factor_node = connecting_factors[0]
        
        # Compute message by evaluating factor at each particle
        particles = particle_node.get_particle_positions()
        weights = particle_node.get_particle_weights()
        
        # For each particle, compute the conditional distribution
        conditional_means = []
        conditional_precisions = []
        
        for i, particle_pos in enumerate(particles):
            # Compute conditional parameters based on factor potential
            if factor_node in self.factor_potentials:
                factor_potential = self.factor_potentials[factor_node]
                factor_precision = factor_potential['precision']
                factor_info = factor_potential['information']
                
                # Assume binary factor connecting particle_node and target_node
                dim = particle_pos.shape[0]
                if factor_precision.shape[0] == 2 * dim:
                    # Determine which block corresponds to which variable
                    neighbors = [n for n in self.graph.neighbors(factor_node) 
                               if not n.startswith('f_') and not n.startswith('obs_')]
                    sorted_neighbors = sorted(neighbors)
                    
                    if particle_node_id == sorted_neighbors[0]:
                        # Particle node is first variable
                        # Condition on particle position for second variable
                        P11 = factor_precision[:dim, :dim]
                        P12 = factor_precision[:dim, dim:2*dim]
                        P22 = factor_precision[dim:2*dim, dim:2*dim]
                        h1 = factor_info[:dim]
                        h2 = factor_info[dim:2*dim]
                        
                        # Conditional: P(x2|x1) ~ N(mu2|1, Sigma2|1)
                        # mu2|1 = P22^{-1} * (h2 - P12^T * x1)
                        # Sigma2|1 = P22^{-1}
                        cond_precision = P22
                        cond_info = h2 - P12.T @ particle_pos
                    else:
                        # Particle node is second variable
                        P11 = factor_precision[:dim, :dim]
                        P12 = factor_precision[:dim, dim:2*dim]
                        P22 = factor_precision[dim:2*dim, dim:2*dim]
                        h1 = factor_info[:dim]
                        h2 = factor_info[dim:2*dim]
                        
                        # Conditional: P(x1|x2) ~ N(mu1|2, Sigma1|2)
                        cond_precision = P11
                        cond_info = h1 - P12 @ particle_pos
                    
                    cond_precision = regularize_precision_matrix(cond_precision)
                    cond_mean = safe_precision_to_mean(cond_precision, cond_info)
                    
                    conditional_means.append(cond_mean)
                    conditional_precisions.append(cond_precision)
        
        if not conditional_means:
            # Fallback to direct Gaussian approximation
            return particle_node.compute_gaussian_approximation()
        
        # Moment matching: approximate mixture of Gaussians with single Gaussian
        conditional_means = np.array(conditional_means)
        
        # Weighted mean of means
        mean_of_means = np.average(conditional_means, weights=weights, axis=0)
        
        # Weighted covariance (including between-component variance)
        total_cov = np.zeros((conditional_means.shape[1], conditional_means.shape[1]))
        for i, (mean_i, prec_i) in enumerate(zip(conditional_means, conditional_precisions)):
            cov_i = np.linalg.inv(prec_i)
            diff_i = mean_i - mean_of_means
            total_cov += weights[i] * (cov_i + np.outer(diff_i, diff_i))
        
        total_cov += 1e-6 * np.eye(total_cov.shape[0])  # Regularization
        total_precision = np.linalg.inv(total_cov)
        
        return GaussianMessage(mean_of_means, total_precision)
    
    def gaussian_to_particle_log_prob_grad(self, particle_node_id: str) -> Callable[[np.ndarray], np.ndarray]:
        """
        Create a function that computes log probability gradients for particle updates.
        
        Returns a function that takes particle positions and returns gradients.
        """
        def log_prob_grad_fn(particles: np.ndarray) -> np.ndarray:
            """
            Compute log probability gradients for each particle.
            
            Args:
                particles: Particle positions, shape (num_particles, dim)
                
            Returns:
                Gradients, shape (num_particles, dim)
            """
            num_particles, dim = particles.shape
            gradients = np.zeros_like(particles)
            
            # Get messages from neighboring nodes
            neighbors = list(self.graph.neighbors(particle_node_id))
            
            for i, particle_pos in enumerate(particles):
                grad_i = np.zeros(dim)
                
                for neighbor in neighbors:
                    if neighbor.startswith('f_') or neighbor.startswith('obs_'):
                        # Factor node - compute gradient from factor potential
                        if neighbor in self.factor_potentials:
                            factor_potential = self.factor_potentials[neighbor]
                            factor_precision = factor_potential['precision']
                            factor_info = factor_potential['information']
                            
                            if factor_precision.shape[0] == dim:
                                # Unary factor: gradient of -0.5 * x^T P x + h^T x
                                grad_i += factor_info - factor_precision @ particle_pos
                            elif factor_precision.shape[0] == 2 * dim:
                                # Binary factor: find particle node's position in factor
                                factor_neighbors = list(self.graph.neighbors(neighbor))
                                if len(factor_neighbors) == 2:
                                    # Determine which position (0 or 1) the particle node occupies
                                    particle_idx = factor_neighbors.index(particle_node_id)
                                    
                                    # Extract relevant 2x2 block from 4x4 precision matrix
                                    start_idx = particle_idx * dim
                                    end_idx = start_idx + dim
                                    P_ii = factor_precision[start_idx:end_idx, start_idx:end_idx]
                                    h_i = factor_info[start_idx:end_idx]
                                    
                                    # For binary factors connecting to other variables,
                                    # we need to consider the other variable's current estimate
                                    other_node = factor_neighbors[1 - particle_idx]
                                    if other_node in self.beliefs:
                                        # Get other node's current belief mean
                                        other_mean = self.beliefs[other_node].mean
                                        
                                        # Extract cross-term precision block
                                        other_start = (1 - particle_idx) * dim
                                        other_end = other_start + dim
                                        P_ij = factor_precision[start_idx:end_idx, other_start:other_end]
                                        
                                        # Gradient includes cross-term with other variable
                                        grad_i += h_i - P_ii @ particle_pos - P_ij @ other_mean
                                    else:
                                        # If other node belief not available, use only diagonal term
                                        grad_i += h_i - P_ii @ particle_pos
                    else:
                        # Variable node - get message and compute gradient
                        msg = self.messages.get((neighbor, particle_node_id))
                        if msg is not None:
                            # Gradient of -0.5 * (x - mu)^T P (x - mu)
                            grad_i += msg.precision @ (msg.mean - particle_pos)
                
                gradients[i] = grad_i
            
            return gradients
        
        return log_prob_grad_fn
    
    def update_particle_beliefs(self):
        """Update beliefs for all particle nodes using SVGD."""
        # print(f"[DEBUG] update_particle_beliefs called - particle_nodes: {list(self.particle_nodes.keys())}")
        for particle_node_id, particle_node in self.particle_nodes.items():
            # print(f"[DEBUG] Updating particle node: {particle_node_id}")
            # Create log probability gradient function
            log_prob_grad_fn = self.gaussian_to_particle_log_prob_grad(particle_node_id)
            
            # Perform SVGD steps
            for _ in range(self.svgd_num_steps):
                self.svgd_optimizer.step(particle_node, log_prob_grad_fn, self.svgd_step_size)
    
    def update_messages_with_particles(self) -> float:
        """
        Perform message passing including particle nodes.
        
        Returns:
            Maximum change in message means
        """
        # First update particle beliefs
        self.update_particle_beliefs()
        
        # Then perform standard message passing, handling particle nodes specially
        new_messages = {}
        max_change = 0.0
        
        # Update all variable-to-factor messages (including particle nodes)
        for edge in self.graph.edges():
            node1, node2 = edge
            
            # Identify factor and variable nodes
            if node1.startswith('f_') or node1.startswith('obs_'):
                factor_node, var_node = node1, node2
            else:
                factor_node, var_node = node2, node1
            
            # Update variable-to-factor message
            old_msg = self.messages.get((var_node, factor_node))
            
            if self.is_particle_node(var_node):
                # Particle-to-factor message
                new_msg = self.particle_to_gaussian_message(var_node, factor_node)
            else:
                # Standard Gaussian-to-factor message
                new_msg = self.variable_to_factor_message(var_node, factor_node)
            
            new_messages[(var_node, factor_node)] = new_msg
            
            if old_msg is not None:
                change = np.linalg.norm(new_msg.mean - old_msg.mean)
                max_change = max(max_change, change)
        
        # Update messages dictionary
        self.messages.update(new_messages)
        
        # Update factor-to-variable messages
        for edge in self.graph.edges():
            node1, node2 = edge
            
            if node1.startswith('f_') or node1.startswith('obs_'):
                factor_node, var_node = node1, node2
            else:
                factor_node, var_node = node2, node1
            
            # Update factor-to-variable message
            old_msg = self.messages.get((factor_node, var_node))
            
            if self.is_particle_node(var_node):
                # Factor-to-particle message (no explicit message, handled in SVGD)
                dim = getattr(self, 'variable_dim', 2)
                new_msg = GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
            else:
                # Standard factor-to-Gaussian message
                new_msg = self.factor_to_variable_message(factor_node, var_node)
            
            new_messages[(factor_node, var_node)] = new_msg
            
            if old_msg is not None:
                change = np.linalg.norm(new_msg.mean - old_msg.mean)
                max_change = max(max_change, change)
        
        # Final update
        self.messages.update(new_messages)
        return max_change
    
    def compute_beliefs_with_particles(self):
        """Compute marginal beliefs for all nodes, including particle nodes."""
        # Compute beliefs for Gaussian nodes
        self.compute_beliefs()
        
        # For particle nodes, the belief is represented by the particles themselves
        for particle_node_id, particle_node in self.particle_nodes.items():
            # Store Gaussian approximation as belief for compatibility
            self.beliefs[particle_node_id] = particle_node.compute_gaussian_approximation()