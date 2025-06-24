#!/usr/bin/env python3
"""
Graph creation utilities for different graph topologies
"""

import numpy as np
import networkx as nx


class GraphFactory:
    """Factory class for creating different types of factor graphs."""
    
    @staticmethod
    def create_graph(graph_type: str, grid_size: int = None) -> tuple[nx.Graph, int, int]:
        """Create graph based on specified type.
        
        Returns:
            tuple: (graph, rows, cols) for grid-based graphs
        """
        if graph_type == "3x3_grid":
            return GraphFactory.create_nxn_grid(3)
        elif graph_type == "100x100_grid":
            return GraphFactory.create_nxn_grid(100)
        elif graph_type == "nxn_grid":
            size = grid_size if grid_size is not None else 10
            return GraphFactory.create_nxn_grid(size)
        elif graph_type == "chain":
            return GraphFactory.create_n_chain(5)
        elif graph_type == "n_chain":
            length = grid_size if grid_size is not None else 5
            return GraphFactory.create_n_chain(length)
        elif graph_type == "simple_chain":
            return GraphFactory.create_simple_chain()
        elif graph_type == "particle_terminal_chain":
            return GraphFactory.create_particle_terminal_chain()
        elif graph_type == "particle_terminal_3x3_grid":
            return GraphFactory.create_particle_terminal_3x3_grid()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    @staticmethod
    def create_nxn_grid(n: int) -> tuple[nx.Graph, int, int]:
        """Create NxN grid factor graph."""
        G = nx.Graph()
        rows, cols = n, n
        
        # Variable nodes
        for i in range(n):
            for j in range(n):
                G.add_node(f'x_{i}_{j}')
        
        # Horizontal factors
        for i in range(n):
            for j in range(n-1):
                factor = f'f_h_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i}_{j+1}', factor)
        
        # Vertical factors
        for i in range(n-1):
            for j in range(n):
                factor = f'f_v_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i+1}_{j}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_n_chain(n: int) -> tuple[nx.Graph, int, int]:
        """Create N-node chain factor graph."""
        G = nx.Graph()
        rows, cols = 1, n
        
        # Variable nodes
        for i in range(n):
            G.add_node(f'x_0_{i}')
        
        # Chain factors
        for i in range(n-1):
            factor = f'f_{i}'
            G.add_node(factor)
            G.add_edge(f'x_0_{i}', factor)
            G.add_edge(f'x_0_{i+1}', factor)
        
        return G, rows, cols
    
    
    @staticmethod
    def create_simple_chain() -> tuple[nx.Graph, int, int]:
        """Create simple 3-node 2-edge chain for debugging."""
        G = nx.Graph()
        rows, cols = 1, 3
        
        # Variable nodes: x_0, x_1, x_2
        for i in range(3):
            G.add_node(f'x_{i}')
        
        # Relative position factors between adjacent nodes
        # f_rel_0_1: relative position factor between x_0 and x_1
        G.add_node('f_rel_0_1')
        G.add_edge('x_0', 'f_rel_0_1')
        G.add_edge('x_1', 'f_rel_0_1')
        
        # f_rel_1_2: relative position factor between x_1 and x_2
        G.add_node('f_rel_1_2')
        G.add_edge('x_1', 'f_rel_1_2')
        G.add_edge('x_2', 'f_rel_1_2')
        
        return G, rows, cols
    
    @staticmethod
    def create_particle_terminal_chain() -> tuple[nx.Graph, int, int]:
        """Create 5-node chain with terminal node (x_4) as particle node."""
        G = nx.Graph()
        rows, cols = 1, 5
        
        # Variable nodes: x_0, x_1, x_2, x_3, x_4
        # x_4 will be converted to a particle node later
        for i in range(5):
            G.add_node(f'x_{i}')
        
        # Relative position factors between adjacent nodes
        for i in range(4):
            factor_name = f'f_rel_{i}_{i+1}'
            G.add_node(factor_name)
            G.add_edge(f'x_{i}', factor_name)
            G.add_edge(f'x_{i+1}', factor_name)
        
        return G, rows, cols
    
    @staticmethod
    def create_particle_terminal_3x3_grid() -> tuple[nx.Graph, int, int]:
        """Create 3x3 grid with terminal node (x_2_2) as particle node."""
        G = nx.Graph()
        rows, cols = 3, 3
        
        # Variable nodes
        for i in range(3):
            for j in range(3):
                G.add_node(f'x_{i}_{j}')
        
        # Horizontal factors
        for i in range(3):
            for j in range(2):
                factor = f'f_h_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i}_{j+1}', factor)
        
        # Vertical factors
        for i in range(2):
            for j in range(3):
                factor = f'f_v_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i+1}_{j}', factor)
        
        return G, rows, cols


class PositionManager:
    """Manages node positions for different graph types."""
    
    def __init__(self, graph_type: str, rows: int, cols: int):
        self.graph_type = graph_type
        self.rows = rows
        self.cols = cols
        self.positions = {}
        self.true_positions = {}
    
    def setup_positions(self, graph: nx.Graph, fix_single_node: bool = False):
        """Setup node positions based on graph type."""
        if self.graph_type in ["3x3_grid", "100x100_grid", "nxn_grid"]:
            self._setup_grid_positions(graph)
        elif self.graph_type in ["chain", "n_chain"]:
            self._setup_chain_positions()
        elif self.graph_type == "simple_chain":
            self._setup_simple_chain_positions()
        elif self.graph_type == "particle_terminal_chain":
            self._setup_particle_terminal_chain_positions()
        elif self.graph_type == "particle_terminal_3x3_grid":
            self._setup_particle_terminal_3x3_grid_positions(graph)
        
        # Initialize estimated positions with noise around GT positions
        self._initialize_estimated_positions(fix_single_node)
    
    def _setup_grid_positions(self, graph: nx.Graph):
        """Setup positions for grid graphs."""
        # Variable positions - only set true positions initially
        for i in range(self.rows):
            for j in range(self.cols):
                self.true_positions[f'x_{i}_{j}'] = np.array([j, self.rows-1-i])
        
        # Factor positions (will be drawn dynamically)
        for node in graph.nodes():
            if node.startswith('f_h_'):
                parts = node.split('_')
                i, j = int(parts[2]), int(parts[3])
                self.positions[node] = np.array([j + 0.5, self.rows-1-i])
            elif node.startswith('f_v_'):
                parts = node.split('_')
                i, j = int(parts[2]), int(parts[3])
                self.positions[node] = np.array([j, self.rows-1-i-0.5])
    
    def _setup_chain_positions(self):
        """Setup positions for chain graph."""
        # Variable positions - only set true positions initially
        for i in range(self.cols):
            self.true_positions[f'x_0_{i}'] = np.array([i, 0])
        
        # Factor positions
        for i in range(self.cols-1):
            self.positions[f'f_{i}'] = np.array([i + 0.5, 0.3])
    
    
    def _setup_simple_chain_positions(self):
        """Setup positions for simple chain graph."""
        # Variable positions - only set true positions initially
        for i in range(3):
            self.true_positions[f'x_{i}'] = np.array([i * 2.0, 0])
        
        # Factor positions
        self.positions['f_rel_0_1'] = np.array([1.0, 0.5])
        self.positions['f_rel_1_2'] = np.array([3.0, 0.5])
    
    def _setup_particle_terminal_chain_positions(self):
        """Setup positions for particle terminal chain graph."""
        # Variable positions - 5 nodes with spacing 1.5 units
        for i in range(5):
            self.true_positions[f'x_{i}'] = np.array([i * 1.5, 0])
        
        # Factor positions - between adjacent nodes, slightly above
        for i in range(4):
            factor_name = f'f_rel_{i}_{i+1}'
            self.positions[factor_name] = np.array([i * 1.5 + 0.75, 0.5])
    
    def _setup_particle_terminal_3x3_grid_positions(self, graph: nx.Graph):
        """Setup positions for particle terminal 3x3 grid graph."""
        # Variable positions - only set true positions initially
        for i in range(self.rows):
            for j in range(self.cols):
                self.true_positions[f'x_{i}_{j}'] = np.array([j, self.rows-1-i])
        
        # Factor positions (will be drawn dynamically)
        for node in graph.nodes():
            if node.startswith('f_h_'):
                parts = node.split('_')
                i, j = int(parts[2]), int(parts[3])
                self.positions[node] = np.array([j + 0.5, self.rows-1-i])
            elif node.startswith('f_v_'):
                parts = node.split('_')
                i, j = int(parts[2]), int(parts[3])
                self.positions[node] = np.array([j, self.rows-1-i-0.5])
    
    def _initialize_estimated_positions(self, fix_single_node: bool = False):
        """Initialize estimated positions around origin (not using GT)."""
        np.random.seed(42)  # For reproducible results
        init_std = 1.0  # Standard deviation for origin-centered initialization
        
        # Get which nodes should be fixed at GT position
        fixed_nodes = []
        if fix_single_node:
            fixed_nodes = self.get_observation_nodes(fix_single_node)
        
        for var_name, true_pos in self.true_positions.items():
            if var_name in fixed_nodes:
                # Fixed nodes start at ground truth position
                self.positions[var_name] = true_pos.copy()
            else:
                # Initialize around origin (0, 0) with Gaussian noise
                self.positions[var_name] = np.random.normal(0, init_std, size=2)
    
    def get_observation_nodes(self, fix_single_node: bool = False) -> list:
        """Get observation nodes based on graph type."""
        if fix_single_node:
            # Fix only one source node
            if self.graph_type == "3x3_grid":
                corners = ['x_1_1']  # Center node
            elif self.graph_type in ["100x100_grid", "nxn_grid"]:
                center = self.rows // 2
                corners = [f'x_{center}_{center}']  # Center node
            elif self.graph_type in ["chain", "n_chain"]:
                corners = ['x_0_0']  # First node
            elif self.graph_type == "simple_chain":
                corners = ['x_0']  # First node only
            elif self.graph_type == "particle_terminal_chain":
                corners = ['x_0']  # First node only
            elif self.graph_type == "particle_terminal_3x3_grid":
                corners = ['x_1_1']  # Center node
            else:
                corners = []
        else:
            # Original multiple observation points
            if self.graph_type == "3x3_grid":
                corners = ['x_0_0', 'x_0_2', 'x_2_0', 'x_2_2']
            elif self.graph_type in ["100x100_grid", "nxn_grid"]:
                last = self.rows - 1
                corners = [f'x_0_0', f'x_0_{last}', f'x_{last}_0', f'x_{last}_{last}']  # Four corners
            elif self.graph_type in ["chain", "n_chain"]:
                middle = self.cols // 2
                corners = [f'x_0_{middle}']  # Middle node only
            elif self.graph_type == "simple_chain":
                corners = ['x_0', 'x_2']  # First and last nodes
            elif self.graph_type == "particle_terminal_chain":
                corners = ['x_0', 'x_3']  # First and second-to-last nodes (x_4 is particle)
            elif self.graph_type == "particle_terminal_3x3_grid":
                corners = ['x_0_0', 'x_0_2', 'x_2_0']  # Three corners (x_2_2 is particle)
            else:
                corners = []
        
        return corners