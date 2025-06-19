#!/usr/bin/env python3
"""
Gaussian Belief Propagation algorithm implementation
"""

import numpy as np
import networkx as nx
from .gaussian_message import GaussianMessage


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
    
    def create_linear_factor(self, measurement: np.ndarray, measurement_precision: np.ndarray,
                           jacobian: np.ndarray, residual_offset: np.ndarray = None) -> tuple:
        """Create factor potential for linear measurement function.
        
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
    
    def create_range_bearing_factor(self, robot1_pos: np.ndarray, robot2_pos: np.ndarray,
                                  measurement: np.ndarray, measurement_precision: np.ndarray) -> tuple:
        """Create factor for range-bearing measurement between two robots.
        
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
        
        return self.create_linear_factor(measurement, measurement_precision, jacobian, residual_offset)
    
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
        
        # Add small regularization to prevent singular matrices
        precision_sum += 1e-8 * np.eye(precision_sum.shape[0])
        
        # Convert back to mean form for message
        try:
            mean = np.linalg.solve(precision_sum, information_sum)
        except np.linalg.LinAlgError:
            # Fallback for singular matrices
            dim = getattr(self, 'variable_dim', 2)
            return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
        return GaussianMessage(mean, precision_sum)
    
    def compute_range_bearing(self, pos1: np.ndarray, pos2: np.ndarray) -> np.ndarray:
        """Compute range-bearing measurement h(x1, x2) = [r, b]"""
        diff = pos2 - pos1
        range_val = np.linalg.norm(diff)
        bearing = np.arctan2(diff[1], diff[0])
        return np.array([range_val, bearing])
    
    def create_prior_factor(self, prior_mean: np.ndarray, prior_precision: np.ndarray) -> tuple:
        """Create a unary prior factor.
        
        Args:
            prior_mean: prior mean μ
            prior_precision: prior precision Λ
            
        Returns:
            (precision_matrix, information_vector)
        """
        # For prior factor: Λ_f = Λ, η_f = Λ * μ
        return prior_precision, prior_precision @ prior_mean
    
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
            try:
                mean = np.linalg.solve(factor_precision, factor_info)
                return GaussianMessage(mean, factor_precision)
            except np.linalg.LinAlgError:
                dim = getattr(self, 'variable_dim', 2)
                return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
        # Case 2: Binary factor (connected to target + one other variable)
        if len(other_vars) == 1:
            other_var = other_vars[0]
            msg_from_other = self.messages.get((other_var, factor_node))
            
            # Determine variable ordering
            all_vars = sorted(connected_vars)
            var_idx = all_vars.index(var_node)
            other_idx = all_vars.index(other_var)
            
            # For NxN precision matrix representing variables
            var_dim = getattr(self, 'variable_dim', 2)
            expected_size = var_dim * 2  # Two variables
            if factor_precision.shape[0] == expected_size:
                # Block structure: [[P11, P12], [P21, P22]]
                P11 = factor_precision[0:var_dim, 0:var_dim]
                P12 = factor_precision[0:var_dim, var_dim:2*var_dim] 
                P21 = factor_precision[var_dim:2*var_dim, 0:var_dim]
                P22 = factor_precision[var_dim:2*var_dim, var_dim:2*var_dim]
                
                eta1 = factor_info[0:var_dim]
                eta2 = factor_info[var_dim:2*var_dim]
                
                # Determine which variable to marginalize out (other_var) and which to keep (var_node)
                if var_idx == 0:  # Target variable is first, marginalize second
                    if msg_from_other is not None:
                        P22_aug = P22 + msg_from_other.precision
                        eta2_aug = eta2 + msg_from_other.precision @ msg_from_other.mean
                    else:
                        P22_aug = P22 + 1e-6 * np.eye(var_dim)
                        eta2_aug = eta2
                    
                    # Keep first block, marginalize second
                    P_keep, P_cross, P_marg = P11, P12, P22_aug
                    eta_keep, eta_marg = eta1, eta2_aug
                    
                else:  # Target variable is second, marginalize first
                    if msg_from_other is not None:
                        P11_aug = P11 + msg_from_other.precision
                        eta1_aug = eta1 + msg_from_other.precision @ msg_from_other.mean
                    else:
                        P11_aug = P11 + 1e-6 * np.eye(var_dim)
                        eta1_aug = eta1
                    
                    # Keep second block, marginalize first 
                    P_keep, P_cross, P_marg = P22, P21, P11_aug
                    eta_keep, eta_marg = eta2, eta1_aug
                
                # Marginalize using Schur complement
                try:
                    P_marg_inv = np.linalg.inv(P_marg)
                    marginal_precision = P_keep - P_cross @ P_marg_inv @ P_cross.T
                    marginal_info = eta_keep - P_cross @ P_marg_inv @ eta_marg
                    
                    # Add regularization
                    marginal_precision += 1e-8 * np.eye(marginal_precision.shape[0])
                    marginal_mean = np.linalg.solve(marginal_precision, marginal_info)
                    
                    return GaussianMessage(marginal_mean, marginal_precision)
                    
                except np.linalg.LinAlgError:
                    # Fallback for numerical issues
                    dim = getattr(self, 'variable_dim', 2)
                    return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
            
            # Fallback for other matrix sizes
            try:
                mean = np.linalg.solve(factor_precision, factor_info)
                return GaussianMessage(mean[:var_dim], factor_precision[:var_dim, :var_dim])
            except np.linalg.LinAlgError:
                dim = getattr(self, 'variable_dim', 2)
                return GaussianMessage(np.zeros(dim), 1e-6 * np.eye(dim))
        
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
                    precision_sum += 1e-8 * np.eye(precision_sum.shape[0])
                    mean = np.linalg.solve(precision_sum, information_sum)
                    self.beliefs[node] = GaussianMessage(mean, precision_sum)
                else:
                    # No messages received - set to weak prior
                    dim = getattr(self, 'variable_dim', 2)
                    weak_precision = 1e-4 * np.eye(dim)
                    self.beliefs[node] = GaussianMessage(np.zeros(dim), weak_precision)