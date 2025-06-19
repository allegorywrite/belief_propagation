#!/usr/bin/env python3
"""
Animation and visualization module for Gaussian BP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from .gaussian_bp import GaussianBP
from .gaussian_message import GaussianMessage
from .graph_utils import GraphFactory, PositionManager


class DynamicEdgeBPAnimation:
    """BP animation with edges connecting estimated positions."""
    
    def __init__(self, graph_type="3x3_grid", fix_single_node=False, 
                 relative_weight=1.0, smoothness_weight=0.1, anchor_weight=10000.0):
        self.graph_type = graph_type
        self.fix_single_node = fix_single_node
        
        # Factor weights
        self.relative_weight = relative_weight      # Weight for relative position factors
        self.smoothness_weight = smoothness_weight  # Weight for smoothness factors  
        self.anchor_weight = anchor_weight          # Weight for anchor factors
        
        self.create_graph()
        self.setup_positions()
        self.setup_bp()
        
        # Animation settings
        self.message_frames = 8
        self.pause_time = 0.15
        
        plt.ion()
        
        # Create two separate windows
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle('Gaussian BP Visualization')
        
        # Convergence plot window - single plot for all errors
        self.conv_fig, self.conv_ax = plt.subplots(1, 1, figsize=(8, 6))
        self.conv_fig.suptitle('Error Convergence Analysis')
        
        # Convergence tracking
        self.iteration_history = []
        self.total_error_history = []
        self.edge_error_history = {}  # Per-edge error tracking
        
    def create_graph(self):
        """Create graph based on specified type."""
        self.G, self.rows, self.cols = GraphFactory.create_graph(self.graph_type)
    
    def setup_positions(self):
        """Setup node positions based on graph type."""
        self.position_manager = PositionManager(self.graph_type, self.rows, self.cols)
        self.position_manager.setup_positions(self.G, self.fix_single_node)
        self.positions = self.position_manager.positions
        self.true_positions = self.position_manager.true_positions
    
    def setup_bp(self):
        """Setup BP with relative position factors and smoothness factors."""
        self.bp = GaussianBP(self.G)
        self.bp.initialize_messages(dim=2)
        
        # Set up both factor types
        self.setup_relative_position_factors()
        self.setup_smoothness_factors()
        if self.fix_single_node:
            self.add_single_anchor()
    
    def setup_priors(self):
        """Set up prior factors for all variables."""
        # Set weak priors for all variables (high uncertainty)
        weak_precision = 0.01 * np.eye(2)  # Very weak
        
        for var_name in self.true_positions.keys():
            prior_name = f'f_prior_{var_name}'
            
            # Add prior factor node if it doesn't exist
            if prior_name not in self.G.nodes():
                self.G.add_node(prior_name)
                self.G.add_edge(var_name, prior_name)
            
            # Use noisy initial position as prior mean
            prior_mean = self.positions[var_name]
            precision, info = self.bp.create_prior_factor(prior_mean, weak_precision)
            self.bp.set_factor_potential(prior_name, precision, info)
    
    def setup_relative_position_factors(self):
        """Set up relative position factors between adjacent variables."""
        # Relative position measurement precision (weighted by relative_weight)
        base_precision = np.array([[1.0, 0.0], [0.0, 1.0]])
        relative_precision = self.relative_weight * base_precision
        
        for factor_node in self.G.nodes():
            if factor_node.startswith('f_rel_') or (factor_node.startswith('f_') and not factor_node.startswith('f_prior')):
                # Get connected variables
                connected_vars = [n for n in self.G.neighbors(factor_node) 
                                if not factor_node.startswith('obs_')]
                
                if len(connected_vars) == 2:
                    var1, var2 = connected_vars
                    
                    # True relative position measurement
                    true_pos1 = self.true_positions.get(var1, np.zeros(2))
                    true_pos2 = self.true_positions.get(var2, np.zeros(2))
                    true_relative = true_pos2 - true_pos1
                    
                    # Create linear factor for relative position constraint
                    # Measurement: z = x2 - x1 (relative position)
                    # Jacobian: [-I, I] (2x4 matrix)
                    jacobian = np.array([[-1.0, 0.0, 1.0, 0.0],
                                       [0.0, -1.0, 0.0, 1.0]])
                    
                    # Measurement is the true relative position (with possible noise)
                    measurement = true_relative
                    
                    precision, info = self.bp.create_linear_factor(
                        measurement, relative_precision, jacobian
                    )
                    self.bp.set_factor_potential(factor_node, precision, info)
    
    def add_single_anchor(self):
        """Add single anchor point for reference frame (only when fix_single_node=True)."""
        corners = self.position_manager.get_observation_nodes(self.fix_single_node)
        
        if corners:
            anchor_node = corners[0]  # Use first node as anchor
            if anchor_node in self.true_positions:
                obs_node = f'obs_{anchor_node}'
                self.G.add_node(obs_node)
                self.G.add_edge(anchor_node, obs_node)
                
                true_pos = self.true_positions[anchor_node]
                
                # Single anchor with very high precision to fix reference frame
                obs_precision = self.anchor_weight * np.eye(2)
                
                precision, info = self.bp.create_prior_factor(true_pos, obs_precision)
                self.bp.set_factor_potential(obs_node, precision, info)
    
    def setup_smoothness_factors(self):
        """Set up smoothness factors that encourage minimal changes in estimated relative positions."""
        # Smoothness precision (weighted by smoothness_weight)
        base_precision = np.array([[1.0, 0.0], [0.0, 1.0]])
        smoothness_precision = self.smoothness_weight * base_precision
        
        # Add smoothness factors for each pair of adjacent variables
        # First collect all relevant factor nodes to avoid changing graph during iteration
        relevant_factors = []
        for factor_node in list(self.G.nodes()):
            if factor_node.startswith('f_rel_') or (factor_node.startswith('f_') and not factor_node.startswith('f_prior') and not factor_node.startswith('obs_')):
                # Get connected variables
                connected_vars = [n for n in self.G.neighbors(factor_node) 
                                if not n.startswith('obs_') and not n.startswith('f_')]
                
                if len(connected_vars) == 2:
                    relevant_factors.append((factor_node, connected_vars))
        
        # Now add smoothness factors
        smoothness_factor_count = 0
        for factor_node, (var1, var2) in relevant_factors:
            # Current estimated relative position (from initial noisy positions)
            current_pos1 = self.positions.get(var1, np.zeros(2))
            current_pos2 = self.positions.get(var2, np.zeros(2))
            estimated_relative = current_pos2 - current_pos1
            
            # Create smoothness factor node
            smoothness_factor = f'f_smooth_{smoothness_factor_count}'
            self.G.add_node(smoothness_factor)
            self.G.add_edge(var1, smoothness_factor)
            self.G.add_edge(var2, smoothness_factor)
            
            # Create linear factor for smoothness constraint
            # Measurement: z = x2 - x1 (maintain current estimated relative position)
            # Jacobian: [-I, I] (2x4 matrix)
            jacobian = np.array([[-1.0, 0.0, 1.0, 0.0],
                               [0.0, -1.0, 0.0, 1.0]])
            
            # Measurement is the current estimated relative position
            measurement = estimated_relative
            
            precision, info = self.bp.create_linear_factor(
                measurement, smoothness_precision, jacobian
            )
            self.bp.set_factor_potential(smoothness_factor, precision, info)
            
            # Set position for smoothness factor (offset from relative position factor)
            if var1 in self.positions and var2 in self.positions:
                pos1 = self.positions[var1]
                pos2 = self.positions[var2]
                # Place smoothness factor slightly offset from midpoint
                self.positions[smoothness_factor] = (pos1 + pos2) / 2 + np.array([0.0, 0.3])
            
            smoothness_factor_count += 1
    
    def get_current_estimated_positions(self):
        """Get current estimated positions of all variables."""
        estimated_positions = {}
        
        # Always use initial positions as fallback, then update with beliefs if available
        for var in self.true_positions.keys():
            # Start with initial noisy position
            estimated_positions[var] = self.positions[var]
            
            # Update with BP belief if available
            if hasattr(self.bp, 'beliefs') and self.bp.beliefs and var in self.bp.beliefs:
                estimated_positions[var] = self.bp.beliefs[var].mean
        
        return estimated_positions
    
    def get_variable_connections(self):
        """Get variable-to-variable connections through factors."""
        connections = []
        
        for factor in self.G.nodes():
            if factor.startswith('f_') and not factor.startswith('f_prior'):
                connected_vars = [n for n in self.G.neighbors(factor) 
                                if n.startswith('x_') or not n.startswith('f_') and not n.startswith('obs_')]
                
                if len(connected_vars) == 2:
                    connections.append((connected_vars[0], connected_vars[1]))
        
        return connections
    
    def compute_edge_errors(self):
        """Compute error for each edge factor."""
        edge_errors = {}
        total_error = 0.0
        
        estimated_positions = self.get_current_estimated_positions()
        
        for factor_node in self.G.nodes():
            if factor_node.startswith('f_rel_') or (factor_node.startswith('f_') and not factor_node.startswith('f_prior') and not factor_node.startswith('obs_') and not factor_node.startswith('f_smooth')):
                # Get connected variables
                connected_vars = [n for n in self.G.neighbors(factor_node) 
                                if not n.startswith('obs_') and not n.startswith('f_')]
                
                if len(connected_vars) == 2:
                    var1, var2 = connected_vars
                    
                    # True relative position
                    true_pos1 = self.true_positions.get(var1, np.zeros(2))
                    true_pos2 = self.true_positions.get(var2, np.zeros(2))
                    true_relative = true_pos2 - true_pos1
                    
                    # Current estimated relative position
                    est_pos1 = estimated_positions.get(var1, np.zeros(2))
                    est_pos2 = estimated_positions.get(var2, np.zeros(2))
                    est_relative = est_pos2 - est_pos1
                    
                    # Error for this edge
                    error = np.linalg.norm(true_relative - est_relative)
                    edge_errors[factor_node] = error
                    total_error += error
        
        return edge_errors, total_error
    
    def update_convergence_plots(self, iteration, message_change):
        """Update convergence plots - single plot with all edge errors and total error."""
        edge_errors, total_error = self.compute_edge_errors()
        
        # Store history
        self.iteration_history.append(iteration)
        self.total_error_history.append(total_error)
        
        # Store per-edge errors
        for edge, error in edge_errors.items():
            if edge not in self.edge_error_history:
                self.edge_error_history[edge] = []
            self.edge_error_history[edge].append(error)
        
        # Clear and update plot
        self.conv_ax.clear()
        
        # Plot per-edge errors with different colors
        colors = ['g', 'r', 'c', 'm', 'y', 'orange', 'purple']
        for i, (edge, errors) in enumerate(self.edge_error_history.items()):
            color = colors[i % len(colors)]
            self.conv_ax.plot(self.iteration_history[-len(errors):], errors, 
                             color=color, label=f'Edge: {edge}', linewidth=1.5, linestyle='--', alpha=0.7)
        
        # Plot total error with prominent style
        self.conv_ax.plot(self.iteration_history, self.total_error_history, 'b-', 
                         label='Total Error', linewidth=3, alpha=0.9)
        
        self.conv_ax.set_xlabel('Iteration')
        self.conv_ax.set_ylabel('Error')
        self.conv_ax.set_title('Per-Edge Errors and Total Error Convergence')
        self.conv_ax.grid(True, alpha=0.3)
        
        # Add legend with total error first
        if self.edge_error_history:
            handles, labels = self.conv_ax.get_legend_handles_labels()
            # Move total error to front
            total_idx = next(i for i, label in enumerate(labels) if 'Total Error' in label)
            handles = [handles[total_idx]] + [handles[i] for i in range(len(handles)) if i != total_idx]
            labels = [labels[total_idx]] + [labels[i] for i in range(len(labels)) if i != total_idx]
            self.conv_ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and refresh
        self.conv_fig.tight_layout()
        self.conv_fig.canvas.draw()
        self.conv_fig.canvas.flush_events()
    
    def draw_dynamic_edges(self, estimated_positions):
        """Draw edges connecting current estimated positions with factor nodes."""
        connections = self.get_variable_connections()
        
        for var1, var2 in connections:
            if var1 in estimated_positions and var2 in estimated_positions:
                pos1 = estimated_positions[var1]
                pos2 = estimated_positions[var2]
                
                # Draw edge between estimated positions
                self.ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           color='gray', linewidth=1.5, alpha=0.6)
                
                # Draw factor node at midpoint of edge
                midpoint = (pos1 + pos2) / 2
                self.ax.plot(midpoint[0], midpoint[1], 's', color='orange', markersize=6,
                           markeredgecolor='darkorange', markeredgewidth=1, alpha=0.8)
    
    def compute_axis_limits(self, estimated_positions, margin_factor=0.15):
        """Compute appropriate axis limits based on node positions.
        
        Args:
            estimated_positions: Dictionary of current estimated positions
            margin_factor: Fraction of range to add as margin
            
        Returns:
            tuple: (x_min, x_max, y_min, y_max)
        """
        # Combine true positions and estimated positions for comprehensive bounds
        all_positions = dict(self.true_positions)
        if estimated_positions:
            all_positions.update(estimated_positions)
            
        if not all_positions:
            # Ultimate fallback
            return -1, 1, -1, 1
        
        # Get all position coordinates
        position_values = list(all_positions.values())
        x_coords = [pos[0] for pos in position_values]
        y_coords = [pos[1] for pos in position_values]
        
        # Compute ranges
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Compute current ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Ensure minimum range for visibility (avoid too zoomed in)
        min_range = 1.0
        if x_range < min_range:
            x_center = (x_min + x_max) / 2
            x_min, x_max = x_center - min_range/2, x_center + min_range/2
            x_range = min_range
            
        if y_range < min_range:
            y_center = (y_min + y_max) / 2
            y_min, y_max = y_center - min_range/2, y_center + min_range/2
            y_range = min_range
        
        # Add margins
        x_margin = max(x_range * margin_factor, 0.2)  # Minimum margin
        y_margin = max(y_range * margin_factor, 0.2)
        
        return (x_min - x_margin, x_max + x_margin, 
                y_min - y_margin, y_max + y_margin)
    
    def draw_base_elements(self, title, estimated_positions):
        """Draw base elements with dynamic edges."""
        self.ax.clear()
        
        # Compute axis limits dynamically based on all node positions
        x_min, x_max, y_min, y_max = self.compute_axis_limits(estimated_positions)
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw dynamic edges connecting estimated positions
        self.draw_dynamic_edges(estimated_positions)
        
        # Draw true positions (black X)
        for var, true_pos in self.true_positions.items():
            self.ax.plot(true_pos[0], true_pos[1], 'kx', markersize=12, markeredgewidth=3)
    
    def draw_current_estimates(self, estimated_positions):
        """Draw current position estimates and uncertainty ellipses."""
        for var in self.true_positions.keys():
            # Always draw all nodes
            if hasattr(self.bp, 'beliefs') and self.bp.beliefs and var in self.bp.beliefs:
                # Node has BP belief - green circle
                belief = self.bp.beliefs[var]
                mean = belief.mean
                cov = belief.covariance
                
                self.ax.plot(mean[0], mean[1], 'o', color='green', markersize=8,
                           alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=1)
                
                # Draw uncertainty ellipse
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
                    
                    # Limit ellipse size for visibility
                    width = min(width, 0.5)
                    height = min(height, 0.5)
                    width = max(width, 0.1)  # Minimum size
                    height = max(height, 0.1)
                    
                    ellipse = Ellipse(mean, width, height, angle=angle,
                                    fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                    self.ax.add_patch(ellipse)
                except Exception as e:
                    # Fallback ellipse if calculation fails
                    ellipse = Ellipse(mean, 0.2, 0.2, angle=0,
                                    fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                    self.ax.add_patch(ellipse)
            else:
                # Node only has initial position - draw with initial uncertainty
                est_pos = estimated_positions.get(var, self.positions[var])
                self.ax.plot(est_pos[0], est_pos[1], 'o', color='green', markersize=8,
                           alpha=0.6, markeredgecolor='darkgreen', markeredgewidth=1)
                
                # Draw initial uncertainty ellipse
                ellipse = Ellipse(est_pos, 0.3, 0.3, angle=0,
                                fill=False, edgecolor='red', linewidth=1.5, alpha=0.5)
                self.ax.add_patch(ellipse)
    
    def animate_message_propagation(self, iteration):
        """Animate messages moving along dynamic edges."""
        estimated_positions = self.get_current_estimated_positions()
        connections = self.get_variable_connections()
        
        for frame in range(self.message_frames):
            progress = frame / (self.message_frames - 1)
            
            self.draw_base_elements(f'Message Propagation - Iteration {iteration}', estimated_positions)
            self.draw_current_estimates(estimated_positions)
            
            # Draw messages moving along edges between estimated positions
            for var1, var2 in connections:
                if var1 in estimated_positions and var2 in estimated_positions:
                    pos1 = estimated_positions[var1]
                    pos2 = estimated_positions[var2]
                    
                    # Message from var1 to var2
                    msg_pos = pos1 + progress * (pos2 - pos1)
                    circle1 = Circle(msg_pos, 0.04, color='blue', alpha=0.8)
                    self.ax.add_patch(circle1)
                    
                    # Message from var2 to var1
                    msg_pos_reverse = pos2 + progress * (pos1 - pos2)
                    circle2 = Circle(msg_pos_reverse, 0.04, color='red', alpha=0.8)
                    self.ax.add_patch(circle2)
            
            # Add status info
            self.ax.text(0.02, 0.98, f'Messages along dynamic edges\nIteration: {iteration}',
                        transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            plt.draw()
            plt.pause(self.pause_time / self.message_frames)
    
    def animate_belief_update(self, iteration):
        """Animate belief updates."""
        for flash in range(3):
            estimated_positions = self.get_current_estimated_positions()
            self.draw_base_elements(f'Updating Beliefs - Iteration {iteration}', estimated_positions)
            
            if flash % 2 == 0:
                self.draw_current_estimates(estimated_positions)
            
            self.ax.text(0.02, 0.98, f'Updating positions\nIteration: {iteration}',
                        transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            
            plt.draw()
            plt.pause(self.pause_time / 3)
    
    def show_iteration_result(self, iteration, error):
        """Show final state of iteration."""
        estimated_positions = self.get_current_estimated_positions()
        self.draw_base_elements(f'Iteration {iteration} Complete', estimated_positions)
        self.draw_current_estimates(estimated_positions)
        
        # Add error information
        info_text = f'Iteration: {iteration}\nTotal Error: {error:.3f}\nEdges follow estimated positions'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.draw()
        plt.pause(self.pause_time * 2)
    
    def show_initial_state(self):
        """Show initial state with all noisy positions before BP starts."""
        estimated_positions = {}
        
        # Use only initial noisy positions (no BP beliefs yet)
        for var in self.true_positions.keys():
            estimated_positions[var] = self.positions[var]
        
        self.draw_base_elements('Initial State - All Nodes with Noisy Positions', estimated_positions)
        
        # Draw all initial positions as blue circles
        for var in self.true_positions.keys():
            pos = self.positions[var]
            self.ax.plot(pos[0], pos[1], 'o', color='blue', markersize=8,
                       alpha=0.7, markeredgecolor='darkblue', markeredgewidth=1)
        
        # Add status info
        self.ax.text(0.02, 0.98, 'Initial noisy positions\n(before BP starts)',
                    transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.draw()
        plt.pause(self.pause_time * 4)  # Show longer
    
    def initialize_bp_beliefs(self):
        """Initialize all variable nodes with weak priors at their initial positions."""
        weak_precision = 1e-4 * np.eye(2)
        
        for var_name in self.true_positions.keys():
            # Initialize belief at the initial noisy position
            initial_pos = self.positions[var_name]
            self.bp.beliefs[var_name] = GaussianMessage(initial_pos, weak_precision)
    
    def run_animation(self):
        """Run the complete animation demonstration."""
        print(f"=== Gaussian BP with Message Propagation ({self.graph_type}) ===")
        print(f"• Graph type: {self.graph_type}")
        print(f"• Variables: {len([n for n in self.G.nodes() if not n.startswith('f_') and not n.startswith('obs_')])}")
        print(f"• Fix single node: {'Yes' if self.fix_single_node else 'No'}")
        print("\nLegend:")
        print("• Green circles: Current estimated positions")
        print("• Red ellipses: Uncertainty")
        print("• Gray lines: Edges between estimated positions")
        print("• Orange squares: Factor nodes (on edge midpoints)")
        print("• Black X: True positions")
        print("• Blue/Red dots: Messages propagating along edges")
        
        # BP is already initialized in setup_bp()
        # Initialize all beliefs with initial positions and moderate uncertainty
        large_precision = 2.0 * np.eye(2)
        for var_name in self.true_positions.keys():
            initial_pos = self.positions[var_name]
            self.bp.beliefs[var_name] = GaussianMessage(initial_pos, large_precision)
        
        # Show initial state
        self.draw_current_state("Initial State", 0, 0.0)
        
        max_iterations = 8
        
        try:
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}")
                
                # Animate message propagation
                self.animate_message_propagation(iteration + 1)
                
                # Update messages
                change = self.bp.update_messages()
                print(f"  Message change: {change:.6f}")
                
                # Debug: Show which factors are active
                if iteration == 0:  # Only show on first iteration
                    print(f"  Active factors:")
                    for factor in self.bp.factor_potentials:
                        precision = self.bp.factor_potentials[factor]['precision']
                        max_precision = np.max(precision)
                        print(f"    {factor}: max_precision={max_precision:.6f}")
                
                # Update beliefs
                self.bp.compute_beliefs()
                
                # Animate belief update
                self.animate_belief_update(iteration + 1)
                
                # Calculate error
                total_error = 0.0
                for var, true_pos in self.true_positions.items():
                    if var in self.bp.beliefs:
                        estimated = self.bp.beliefs[var].mean
                        error = np.linalg.norm(estimated - true_pos)
                        total_error += error
                
                print(f"  Total error: {total_error:.3f}")
                
                # Update convergence plots
                self.update_convergence_plots(iteration + 1, change)
                
                # Show iteration result
                self.show_iteration_result(iteration + 1, total_error)
                
                # Check convergence
                # if change < 1e-6:
                #     print("  Converged!")
                #     break
            
            print("Animation complete! Close window to exit.")
            plt.show(block=True)
            
        except KeyboardInterrupt:
            print("\nAnimation interrupted by user")
            plt.show(block=True)
    
    def draw_current_state(self, title, iteration, error):
        """Draw current state with all nodes and uncertainties."""
        # Get current estimated positions
        estimated_positions = {}
        for var in self.true_positions.keys():
            if var in self.bp.beliefs:
                estimated_positions[var] = self.bp.beliefs[var].mean
            else:
                estimated_positions[var] = self.positions[var]
        
        # Clear and redraw
        self.draw_base_elements(title, estimated_positions)
        
        # Draw all nodes and ellipses
        for var in self.true_positions.keys():
            if var in self.bp.beliefs:
                belief = self.bp.beliefs[var]
                pos = belief.mean
                cov = belief.covariance
                
                # Draw node
                self.ax.plot(pos[0], pos[1], 'o', color='green', markersize=8,
                           alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=1)
                
                # Draw uncertainty ellipse
                try:
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
                    
                    # Limit ellipse size
                    width = min(width, 0.5)
                    height = min(height, 0.5)
                    
                    ellipse = Ellipse(pos, width, height, angle=angle,
                                    fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                    self.ax.add_patch(ellipse)
                except:
                    pass
        
        # Add info
        info_text = f'Iteration: {iteration}\nError: {error:.3f}'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.draw()
        plt.pause(1.0)  # Pause to see each iteration