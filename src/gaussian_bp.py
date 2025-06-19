#!/usr/bin/env python3
"""
Gaussian Belief Propagation algorithm implementation
"""

import numpy as np
import networkx as nx
from .gaussian_message import GaussianMessage
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
    """Gaussian Belief Propagation implementation."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.messages = {}
        self.beliefs = {}
        self.factor_potentials = {}
        
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