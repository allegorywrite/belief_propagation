#!/usr/bin/env python3
"""
Belief Propagation problem setup utilities.
Contains logic for setting up BP problems with different factor types.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from ..gaussian_message import GaussianMessage
from .factor_utils import (
    create_prior_factor, 
    create_relative_position_factor,
    create_smoothness_factor
)
from .graph_utils import PositionManager


class BPProblemSetup:
    """Handles setup of BP problems with various factor configurations."""
    
    def __init__(self, graph: nx.Graph, position_manager: PositionManager, bp=None):
        self.graph = graph
        self.position_manager = position_manager
        self.bp = bp  # BP instance will be set externally
        
        # Factor weights
        self.relative_weight = 1.0
        self.smoothness_weight = 0.1
        self.anchor_weight = 10000.0
        
    def set_factor_weights(self, relative_weight: float = 1.0, 
                          smoothness_weight: float = 0.1,
                          anchor_weight: float = 10000.0):
        """Set weights for different factor types."""
        self.relative_weight = relative_weight
        self.smoothness_weight = smoothness_weight
        self.anchor_weight = anchor_weight
    
    def initialize_bp(self, dim: int = 2):
        """Initialize BP with specified dimension."""
        self.bp.initialize_messages(dim=dim)
        return self.bp
    
    def setup_relative_position_factors(self):
        """Set up relative position factors between adjacent variables."""
        base_precision = np.array([[1.0, 0.0], [0.0, 1.0]])
        relative_precision = self.relative_weight * base_precision
        
        for factor_node in self.graph.nodes():
            if factor_node.startswith('f_rel_') or (
                factor_node.startswith('f_') and 
                not factor_node.startswith('f_prior') and
                not factor_node.startswith('f_smooth')
            ):
                # Get connected variables
                connected_vars = [n for n in self.graph.neighbors(factor_node) 
                                if not n.startswith('obs_')]
                
                if len(connected_vars) == 2:
                    var1, var2 = connected_vars
                    
                    # Get true positions
                    true_pos1 = self.position_manager.true_positions.get(var1, np.zeros(2))
                    true_pos2 = self.position_manager.true_positions.get(var2, np.zeros(2))
                    
                    # Create relative position factor
                    precision, info = create_relative_position_factor(
                        true_pos1, true_pos2, relative_precision
                    )
                    self.bp.set_factor_potential(factor_node, precision, info)
    
    def setup_smoothness_factors(self):
        """Set up smoothness factors that encourage minimal changes in estimated relative positions."""
        base_precision = np.array([[1.0, 0.0], [0.0, 1.0]])
        smoothness_precision = self.smoothness_weight * base_precision
        
        # Add smoothness factors for each pair of adjacent variables
        relevant_factors = []
        for factor_node in list(self.graph.nodes()):
            if factor_node.startswith('f_rel_') or (
                factor_node.startswith('f_') and 
                not factor_node.startswith('f_prior') and 
                not factor_node.startswith('obs_') and
                not factor_node.startswith('f_smooth')
            ):
                # Get connected variables
                connected_vars = [n for n in self.graph.neighbors(factor_node) 
                                if not n.startswith('obs_') and not n.startswith('f_')]
                
                if len(connected_vars) == 2:
                    relevant_factors.append((factor_node, connected_vars))
        
        # Now add smoothness factors
        smoothness_factor_count = 0
        for factor_node, (var1, var2) in relevant_factors:
            # Current estimated relative position (from initial noisy positions)
            current_pos1 = self.position_manager.positions.get(var1, np.zeros(2))
            current_pos2 = self.position_manager.positions.get(var2, np.zeros(2))
            
            # Create smoothness factor node
            smoothness_factor = f'f_smooth_{smoothness_factor_count}'
            self.graph.add_node(smoothness_factor)
            self.graph.add_edge(var1, smoothness_factor)
            self.graph.add_edge(var2, smoothness_factor)
            
            # Create smoothness factor
            precision, info = create_smoothness_factor(
                current_pos1, current_pos2, smoothness_precision
            )
            self.bp.set_factor_potential(smoothness_factor, precision, info)
            
            # Set position for smoothness factor (offset from relative position factor)
            if var1 in self.position_manager.positions and var2 in self.position_manager.positions:
                pos1 = self.position_manager.positions[var1]
                pos2 = self.position_manager.positions[var2]
                # Place smoothness factor slightly offset from midpoint
                self.position_manager.positions[smoothness_factor] = (pos1 + pos2) / 2 + np.array([0.0, 0.3])
            
            smoothness_factor_count += 1
    
    def add_single_anchor(self, fix_single_node: bool = False):
        """Add single anchor point for reference frame."""
        if not fix_single_node:
            return
            
        corners = self.position_manager.get_observation_nodes(fix_single_node)
        
        if corners:
            anchor_node = corners[0]  # Use first node as anchor
            if anchor_node in self.position_manager.true_positions:
                obs_node = f'obs_{anchor_node}'
                self.graph.add_node(obs_node)
                self.graph.add_edge(anchor_node, obs_node)
                
                true_pos = self.position_manager.true_positions[anchor_node]
                
                # Single anchor with very high precision to fix reference frame
                obs_precision = self.anchor_weight * np.eye(2)
                
                precision, info = create_prior_factor(true_pos, obs_precision)
                self.bp.set_factor_potential(obs_node, precision, info)
    
    def setup_priors(self):
        """Set up prior factors for all variables."""
        # Set weak priors for all variables (high uncertainty)
        weak_precision = 0.01 * np.eye(2)  # Very weak
        
        for var_name in self.position_manager.true_positions.keys():
            prior_name = f'f_prior_{var_name}'
            
            # Add prior factor node if it doesn't exist
            if prior_name not in self.graph.nodes():
                self.graph.add_node(prior_name)
                self.graph.add_edge(var_name, prior_name)
            
            # Use noisy initial position as prior mean
            prior_mean = self.position_manager.positions[var_name]
            precision, info = create_prior_factor(prior_mean, weak_precision)
            self.bp.set_factor_potential(prior_name, precision, info)
    
    def initialize_beliefs(self):
        """Initialize all variable nodes with beliefs at their initial positions."""
        large_precision = 2.0 * np.eye(2)
        
        for var_name in self.position_manager.true_positions.keys():
            initial_pos = self.position_manager.positions[var_name]
            self.bp.beliefs[var_name] = GaussianMessage(initial_pos, large_precision)
    
    def setup_complete_problem(self, fix_single_node: bool = False, 
                             include_smoothness: bool = True,
                             include_priors: bool = False):
        """
        Set up complete BP problem with all factor types.
        
        Args:
            fix_single_node: Whether to add anchor factor for single node
            include_smoothness: Whether to include smoothness factors
            include_priors: Whether to include prior factors
        """
        # Initialize BP
        self.initialize_bp(dim=2)
        
        # Set up relative position factors
        self.setup_relative_position_factors()
        
        # Add smoothness factors if requested
        if include_smoothness:
            self.setup_smoothness_factors()
        
        # Add anchor if requested
        if fix_single_node:
            self.add_single_anchor(fix_single_node)
        
        # Add priors if requested
        if include_priors:
            self.setup_priors()
        
        # Initialize beliefs
        self.initialize_beliefs()
        
        return self.bp