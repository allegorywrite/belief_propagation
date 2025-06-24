#!/usr/bin/env python3
"""
Visualization and animation module for Gaussian BP.
Contains only visualization/animation logic, algorithms are in separate modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from ..utils.bp_problem_setup import BPProblemSetup
from ..utils.graph_utils import GraphFactory, PositionManager
from ..gaussian_bp import GaussianBP


class DynamicEdgeBPAnimation:
    """BP animation with edges connecting estimated positions."""
    
    def __init__(self, graph_type="3x3_grid", fix_single_node=False, 
                 relative_weight=1.0, smoothness_weight=0.1, anchor_weight=10000.0,
                 particle_nodes=None, no_message_animation=False, 
                 no_variance_display=False, no_error_window=False, grid_size=None):
        self.graph_type = graph_type
        self.fix_single_node = fix_single_node
        self.grid_size = grid_size
        
        # Factor weights
        self.relative_weight = relative_weight
        self.smoothness_weight = smoothness_weight
        self.anchor_weight = anchor_weight
        
        # Particle node configuration
        self.particle_node_config = particle_nodes or self._get_default_particle_config()
        
        # Animation control options
        self.no_message_animation = no_message_animation
        self.no_variance_display = no_variance_display  
        self.no_error_window = no_error_window
        
        self.create_graph()
        self.setup_positions()
        self.setup_bp()
        
        # Animation settings
        self.message_frames = 8
        self.pause_time = 0.15
        
        plt.ion()
        
        # Create main visualization window
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle('Gaussian BP Visualization')
        
        # Convergence plot window - single plot for all errors (only if not suppressed)
        if not self.no_error_window:
            self.conv_fig, self.conv_ax = plt.subplots(1, 1, figsize=(8, 6))
            self.conv_fig.suptitle('Error Convergence Analysis')
        else:
            self.conv_fig, self.conv_ax = None, None
        
        # Convergence tracking
        self.iteration_history = []
        self.total_error_history = []
        self.edge_error_history = {}  # Per-edge error tracking
    
    def _get_default_particle_config(self):
        """Get default particle node configuration for different graph types."""
        defaults = {
            "particle_terminal_chain": {
                "x_4": {"num_particles": 50, "dim": 2, "noise_scale": 1.0}
            },
            "particle_terminal_3x3_grid": {
                "x_2_2": {"num_particles": 50, "dim": 2, "noise_scale": 1.0}
            }
        }
        return defaults.get(self.graph_type, {})
    
    def _setup_particle_nodes(self):
        """Setup particle nodes based on configuration."""
        for node_id, config in self.particle_node_config.items():
            num_particles = config.get('num_particles', 50)
            dim = config.get('dim', 2)
            noise_scale = config.get('noise_scale', 1.0)  # Increased for origin-centered init
            
            # Initialize particles around origin (0, 0) instead of GT
            initial_particles = np.random.normal(
                loc=0.0,  # Origin-centered initialization
                scale=noise_scale, 
                size=(num_particles, dim)
            )
            
            self.bp.add_particle_node(node_id, num_particles, dim, initial_particles)
    
    def _has_particle_nodes(self):
        """Check if the BP instance has any particle nodes."""
        return hasattr(self.bp, 'particle_nodes') and len(self.bp.particle_nodes) > 0
        
    def create_graph(self):
        """Create graph based on specified type."""
        self.G, self.rows, self.cols = GraphFactory.create_graph(self.graph_type, self.grid_size)
    
    def setup_positions(self):
        """Setup node positions based on graph type."""
        self.position_manager = PositionManager(self.graph_type, self.rows, self.cols)
        self.position_manager.setup_positions(self.G, self.fix_single_node)
        self.positions = self.position_manager.positions
        self.true_positions = self.position_manager.true_positions
    
    def setup_bp(self):
        """Setup BP using the problem setup module."""
        # Create BP instance first
        self.bp = GaussianBP(self.G)
        
        # Create problem setup with BP instance
        self.problem_setup = BPProblemSetup(self.G, self.position_manager, self.bp)
        self.problem_setup.set_factor_weights(
            self.relative_weight, 
            self.smoothness_weight, 
            self.anchor_weight
        )
        
        # Add particle nodes based on configuration
        self._setup_particle_nodes()
        
        # Setup complete problem
        self.bp = self.problem_setup.setup_complete_problem(
            fix_single_node=self.fix_single_node,
            include_smoothness=True,
            include_priors=False
        )
    
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
        if self.no_error_window or self.conv_ax is None:
            return
            
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
                           color='navy', linewidth=1.5, alpha=0.8)
                
                # Draw factor node at midpoint of edge
                midpoint = (pos1 + pos2) / 2
                self.ax.plot(midpoint[0], midpoint[1], 's', color='lightblue', markersize=16,
                           markeredgecolor='navy', markeredgewidth=1, alpha=0.8)
    
    def compute_axis_limits(self, estimated_positions, margin_factor=0.15):
        """Compute appropriate axis limits based on node positions."""
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
            # Check if this is a particle node
            if hasattr(self.bp, 'particle_nodes') and var in self.bp.particle_nodes:
                self.draw_particle_node(var)
            # Check if node has BP belief
            elif hasattr(self.bp, 'beliefs') and self.bp.beliefs and var in self.bp.beliefs:
                # Node has BP belief - draw as Gaussian node
                belief = self.bp.beliefs[var]
                mean = belief.mean
                cov = belief.covariance
                
                self.ax.plot(mean[0], mean[1], 'o', color='lightblue', markersize=16,
                           alpha=0.8, markeredgecolor='navy', markeredgewidth=1)
                
                # Draw uncertainty ellipse (only if not suppressed)
                if not self.no_variance_display:
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
                                        fill=True, facecolor='lightblue', edgecolor='none', alpha=0.4)
                        self.ax.add_patch(ellipse)
                    except Exception as e:
                        # Fallback ellipse if calculation fails
                        ellipse = Ellipse(mean, 0.2, 0.2, angle=0,
                                        fill=True, facecolor='lightblue', edgecolor='none', alpha=0.4)
                        self.ax.add_patch(ellipse)
            else:
                # Node only has initial position - draw with initial uncertainty
                est_pos = estimated_positions.get(var, self.positions[var])
                self.ax.plot(est_pos[0], est_pos[1], 'o', color='lightblue', markersize=16,
                           alpha=0.6, markeredgecolor='navy', markeredgewidth=1)
                
                # Draw initial uncertainty ellipse (only if not suppressed)
                if not self.no_variance_display:
                    ellipse = Ellipse(est_pos, 0.3, 0.3, angle=0,
                                    fill=True, facecolor='lightblue', edgecolor='none', alpha=0.4)
                    self.ax.add_patch(ellipse)
    
    def draw_particle_node(self, var):
        """Draw particle node with individual particles and Gaussian approximation."""
        particle_node = self.bp.particle_nodes[var]
        particles = particle_node.get_particle_positions()
        weights = particle_node.get_particle_weights()
        
        # Draw individual particles with size and color based on weights
        # Normalize weights for visualization (scale particle sizes)
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        weight_range = max_weight - min_weight
        
        if weight_range > 0:
            # Normalize weights to [0.2, 1.0] for size scaling
            normalized_weights = 0.2 + 0.8 * (weights - min_weight) / weight_range
            # Scale sizes based on weights (larger particles for higher weights)
            sizes = 50 * normalized_weights
            # Color intensity based on weights
            colors = weights / max_weight if max_weight > 0 else np.ones_like(weights)
        else:
            # All weights are equal
            sizes = np.full_like(weights, 20)
            colors = np.ones_like(weights)
        
        # Draw particles with weight-based visualization
        scatter = self.ax.scatter(particles[:, 0], particles[:, 1], 
                                c=colors, s=sizes, alpha=0.7, marker='o',
                                cmap='Reds', edgecolors='darkred', linewidths=0.5)
        
        # Get Gaussian approximation for this particle node
        gaussian_approx = particle_node.compute_gaussian_approximation()
        mean = gaussian_approx.mean
        cov = gaussian_approx.covariance
        
        # Draw mean position as larger marker (different from Gaussian nodes)
        self.ax.plot(mean[0], mean[1], 'o', color='orange', markersize=18,
                   alpha=0.9, markeredgecolor='darkred', markeredgewidth=2)
        
        # Draw uncertainty ellipse for Gaussian approximation (only if not suppressed)
        if not self.no_variance_display:
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
                
                # Limit ellipse size for visibility
                width = min(width, 0.8)
                height = min(height, 0.8)
                width = max(width, 0.1)  # Minimum size
                height = max(height, 0.1)
                
                ellipse = Ellipse(mean, width, height, angle=angle,
                                fill=True, facecolor='orange', edgecolor='darkred', 
                                alpha=0.3, linewidth=1)
                self.ax.add_patch(ellipse)
            except Exception as e:
                # Fallback ellipse if calculation fails
                ellipse = Ellipse(mean, 0.2, 0.2, angle=0,
                                fill=True, facecolor='orange', edgecolor='darkred', 
                                alpha=0.3, linewidth=1)
                self.ax.add_patch(ellipse)
    
    def animate_message_propagation(self, iteration):
        """Animate messages moving along dynamic edges."""
        if self.no_message_animation:
            # Skip animation, just draw current state briefly
            estimated_positions = self.get_current_estimated_positions()
            self.draw_base_elements(f'Message Propagation - Iteration {iteration}', estimated_positions)
            self.draw_current_estimates(estimated_positions)
            plt.draw()
            plt.pause(0.1)  # Brief pause
            return
            
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
                self.ax.plot(pos[0], pos[1], 'o', color='lightblue', markersize=16,
                           alpha=0.8, markeredgecolor='navy', markeredgewidth=1)
                
                # Draw uncertainty ellipse (only if not suppressed)
                if not self.no_variance_display:
                    try:
                        eigenvals, eigenvecs = np.linalg.eigh(cov)
                        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                        width, height = 2 * np.sqrt(eigenvals)  # 1-sigma ellipse
                        
                        # Limit ellipse size
                        width = min(width, 0.5)
                        height = min(height, 0.5)
                        
                        ellipse = Ellipse(pos, width, height, angle=angle,
                                        fill=True, facecolor='lightblue', edgecolor='none', alpha=0.4)
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
    
    def run_animation(self):
        """Run the complete animation demonstration."""
        print(f"=== Gaussian BP with Message Propagation ({self.graph_type}) ===")
        print(f"• Graph type: {self.graph_type}")
        print(f"• Variables: {len([n for n in self.G.nodes() if not n.startswith('f_') and not n.startswith('obs_')])}")
        print(f"• Fix single node: {'Yes' if self.fix_single_node else 'No'}")
        print("\nLegend:")
        print("• Light blue circles with navy edges: Current estimated positions")
        print("• Light blue filled ellipses: Uncertainty")
        print("• Navy lines: Edges between estimated positions")
        print("• Light blue squares with navy edges: Factor nodes (on edge midpoints)")
        print("• Black X: True positions")
        print("• Blue/Red dots: Messages propagating along edges")
        
        # Show initial state
        self.draw_current_state("Initial State", 0, 0.0)
        
        max_iterations = 20
        
        try:
            for iteration in range(max_iterations):
                print(f"\nIteration {iteration + 1}")
                
                # Animate message propagation
                self.animate_message_propagation(iteration + 1)
                
                # Update messages (use particle-aware version if needed)
                if self._has_particle_nodes():
                    change = self.bp.update_messages_with_particles()
                    print(f"  Message change (with particles): {change:.6f}")
                else:
                    change = self.bp.update_messages()
                    print(f"  Message change: {change:.6f}")
                
                # Debug: Show which factors are active
                if iteration == 0:  # Only show on first iteration
                    print(f"  Active factors:")
                    for factor in self.bp.factor_potentials:
                        precision = self.bp.factor_potentials[factor]['precision']
                        max_precision = np.max(precision)
                        print(f"    {factor}: max_precision={max_precision:.6f}")
                    
                    # Show particle nodes if present
                    if hasattr(self.bp, 'particle_nodes') and self.bp.particle_nodes:
                        print(f"  Particle nodes:")
                        for node_id, particle_node in self.bp.particle_nodes.items():
                            print(f"    {node_id}: {particle_node.num_particles} particles")
                
                # Update beliefs (use particle-aware version if needed)
                if self._has_particle_nodes():
                    self.bp.compute_beliefs_with_particles()
                else:
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
            
            print("Animation complete! Close window to exit.")
            plt.show(block=True)
            
        except KeyboardInterrupt:
            print("\nAnimation interrupted by user")
            plt.show(block=True)
    
    def run_without_animation(self, max_iterations=8, convergence_threshold=1e-6):
        """Run belief propagation without visual animation."""
        print(f"=== Running Gaussian BP ({self.graph_type}) ===")
        print(f"• Graph type: {self.graph_type}")
        print(f"• Variables: {len([n for n in self.G.nodes() if not n.startswith('f_') and not n.startswith('obs_')])}")
        print(f"• Fix single node: {'Yes' if self.fix_single_node else 'No'}")
        
        # Initialize tracking
        iteration_history = []
        error_history = []
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}")
            
            # Update messages (same logic as animated version)
            if self._has_particle_nodes():
                change = self.bp.update_messages_with_particles()
                print(f"  Message change (with particles): {change:.6f}")
            else:
                change = self.bp.update_messages()
                print(f"  Message change: {change:.6f}")
            
            # Update beliefs (same logic as animated version)
            if self._has_particle_nodes():
                self.bp.compute_beliefs_with_particles()
            else:
                self.bp.compute_beliefs()
            
            # Calculate error (same as animated version)
            total_error = 0.0
            for var, true_pos in self.true_positions.items():
                if var in self.bp.beliefs:
                    estimated = self.bp.beliefs[var].mean
                    error = np.linalg.norm(estimated - true_pos)
                    total_error += error
            
            print(f"  Total error: {total_error:.3f}")
            
            # Store history
            iteration_history.append(iteration + 1)
            error_history.append(total_error)
            
            # Check convergence
            if change < convergence_threshold:
                print(f"  Converged at iteration {iteration + 1}")
                break
        
        print(f"\nFinal Results:")
        print(f"• Total iterations: {len(iteration_history)}")
        print(f"• Final error: {error_history[-1]:.6f}")
        print(f"• Final message change: {change:.6f}")
        
        # Return results
        return {
            'iterations': iteration_history,
            'errors': error_history,
            'final_beliefs': self.bp.beliefs.copy() if hasattr(self.bp, 'beliefs') else {},
            'converged': change < convergence_threshold
        }