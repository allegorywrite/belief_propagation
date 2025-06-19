#!/usr/bin/env python3
"""
Graph creation utilities for different graph topologies
"""

import numpy as np
import networkx as nx


class GraphFactory:
    """Factory class for creating different types of factor graphs."""
    
    @staticmethod
    def create_graph(graph_type: str) -> tuple[nx.Graph, int, int]:
        """Create graph based on specified type.
        
        Returns:
            tuple: (graph, rows, cols) for grid-based graphs
        """
        if graph_type == "3x3_grid":
            return GraphFactory.create_3x3_grid()
        elif graph_type == "2x3_grid":
            return GraphFactory.create_2x3_grid()
        elif graph_type == "4x4_grid":
            return GraphFactory.create_4x4_grid()
        elif graph_type == "chain":
            return GraphFactory.create_chain()
        elif graph_type == "vertical_chain":
            return GraphFactory.create_vertical_chain()
        elif graph_type == "star":
            return GraphFactory.create_star()
        elif graph_type == "triangle":
            return GraphFactory.create_triangle()
        elif graph_type == "binary_tree":
            return GraphFactory.create_binary_tree()
        elif graph_type == "path_tree":
            return GraphFactory.create_path_tree()
        elif graph_type == "branching_tree":
            return GraphFactory.create_branching_tree()
        elif graph_type == "simple_chain":
            return GraphFactory.create_simple_chain()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    
    @staticmethod
    def create_3x3_grid() -> tuple[nx.Graph, int, int]:
        """Create 3x3 grid factor graph."""
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
    
    @staticmethod
    def create_2x3_grid() -> tuple[nx.Graph, int, int]:
        """Create 2x3 grid factor graph."""
        G = nx.Graph()
        rows, cols = 2, 3
        
        # Variable nodes
        for i in range(2):
            for j in range(3):
                G.add_node(f'x_{i}_{j}')
        
        # Horizontal factors
        for i in range(2):
            for j in range(2):
                factor = f'f_h_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i}_{j+1}', factor)
        
        # Vertical factors
        for i in range(1):
            for j in range(3):
                factor = f'f_v_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i+1}_{j}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_4x4_grid() -> tuple[nx.Graph, int, int]:
        """Create 4x4 grid factor graph."""
        G = nx.Graph()
        rows, cols = 4, 4
        
        # Variable nodes
        for i in range(4):
            for j in range(4):
                G.add_node(f'x_{i}_{j}')
        
        # Horizontal factors
        for i in range(4):
            for j in range(3):
                factor = f'f_h_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i}_{j+1}', factor)
        
        # Vertical factors
        for i in range(3):
            for j in range(4):
                factor = f'f_v_{i}_{j}'
                G.add_node(factor)
                G.add_edge(f'x_{i}_{j}', factor)
                G.add_edge(f'x_{i+1}_{j}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_chain() -> tuple[nx.Graph, int, int]:
        """Create chain factor graph."""
        G = nx.Graph()
        rows, cols = 1, 5
        
        # Variable nodes
        for i in range(cols):
            G.add_node(f'x_0_{i}')
        
        # Chain factors
        for i in range(cols-1):
            factor = f'f_{i}'
            G.add_node(factor)
            G.add_edge(f'x_0_{i}', factor)
            G.add_edge(f'x_0_{i+1}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_vertical_chain() -> tuple[nx.Graph, int, int]:
        """Create vertical chain factor graph."""
        G = nx.Graph()
        rows, cols = 5, 1
        
        # Variable nodes
        for i in range(5):
            G.add_node(f'x_{i}_0')
        
        # Chain factors
        for i in range(4):
            factor = f'f_{i}'
            G.add_node(factor)
            G.add_edge(f'x_{i}_0', factor)
            G.add_edge(f'x_{i+1}_0', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_star() -> tuple[nx.Graph, int, int]:
        """Create star factor graph."""
        G = nx.Graph()
        rows, cols = 3, 3
        
        # Central node
        G.add_node('x_1_1')
        
        # Surrounding nodes
        positions = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for idx, (i, j) in enumerate(positions):
            G.add_node(f'x_{i}_{j}')
            factor = f'f_{idx}'
            G.add_node(factor)
            G.add_edge('x_1_1', factor)
            G.add_edge(f'x_{i}_{j}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_triangle() -> tuple[nx.Graph, int, int]:
        """Create triangle factor graph."""
        G = nx.Graph()
        rows, cols = 2, 2
        
        # Triangle nodes
        nodes = ['x_0_0', 'x_0_1', 'x_1_0']
        for node in nodes:
            G.add_node(node)
        
        # Triangle factors
        edges = [('x_0_0', 'x_0_1'), ('x_0_1', 'x_1_0'), ('x_1_0', 'x_0_0')]
        for idx, (node1, node2) in enumerate(edges):
            factor = f'f_{idx}'
            G.add_node(factor)
            G.add_edge(node1, factor)
            G.add_edge(node2, factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_binary_tree() -> tuple[nx.Graph, int, int]:
        r"""Create binary tree factor graph.
        
        Tree structure:
                x_0_1 (root)
               /     \
           x_1_0   x_1_2
          /   \   /   \
      x_2_0 x_2_1 x_2_2 x_2_3 (leaves)
        """
        G = nx.Graph()
        rows, cols = 3, 4
        
        # Level 0: Root
        G.add_node('x_0_1')
        
        # Level 1: Two children
        G.add_node('x_1_0')
        G.add_node('x_1_2')
        
        # Level 2: Four leaves
        for i in range(4):
            G.add_node(f'x_2_{i}')
        
        # Factors connecting levels
        # Root to level 1
        factor_names = []
        G.add_node('f_root_left')
        G.add_edge('x_0_1', 'f_root_left')
        G.add_edge('x_1_0', 'f_root_left')
        factor_names.append('f_root_left')
        
        G.add_node('f_root_right')
        G.add_edge('x_0_1', 'f_root_right')
        G.add_edge('x_1_2', 'f_root_right')
        factor_names.append('f_root_right')
        
        # Level 1 to level 2
        G.add_node('f_left_0')
        G.add_edge('x_1_0', 'f_left_0')
        G.add_edge('x_2_0', 'f_left_0')
        factor_names.append('f_left_0')
        
        G.add_node('f_left_1')
        G.add_edge('x_1_0', 'f_left_1')
        G.add_edge('x_2_1', 'f_left_1')
        factor_names.append('f_left_1')
        
        G.add_node('f_right_2')
        G.add_edge('x_1_2', 'f_right_2')
        G.add_edge('x_2_2', 'f_right_2')
        factor_names.append('f_right_2')
        
        G.add_node('f_right_3')
        G.add_edge('x_1_2', 'f_right_3')
        G.add_edge('x_2_3', 'f_right_3')
        factor_names.append('f_right_3')
        
        return G, rows, cols
    
    @staticmethod
    def create_path_tree() -> tuple[nx.Graph, int, int]:
        """Create path tree (long chain with branches).
        
        Structure:
        x_0_0 - x_0_1 - x_0_2 - x_0_3 - x_0_4
                 |       |       |
               x_1_1   x_1_2   x_1_3
        """
        G = nx.Graph()
        rows, cols = 2, 5
        
        # Main path
        main_nodes = []
        for i in range(5):
            node = f'x_0_{i}'
            G.add_node(node)
            main_nodes.append(node)
        
        # Branch nodes
        branch_nodes = []
        for i in [1, 2, 3]:  # Branches at positions 1, 2, 3
            node = f'x_1_{i}'
            G.add_node(node)
            branch_nodes.append(node)
        
        # Main path factors
        for i in range(4):
            factor = f'f_main_{i}'
            G.add_node(factor)
            G.add_edge(f'x_0_{i}', factor)
            G.add_edge(f'x_0_{i+1}', factor)
        
        # Branch factors
        for i, pos in enumerate([1, 2, 3]):
            factor = f'f_branch_{pos}'
            G.add_node(factor)
            G.add_edge(f'x_0_{pos}', factor)
            G.add_edge(f'x_1_{pos}', factor)
        
        return G, rows, cols
    
    @staticmethod
    def create_branching_tree() -> tuple[nx.Graph, int, int]:
        r"""Create multi-branch tree.
        
        Structure:
                x_1_2 (root)
            /    |    |    \
        x_0_0  x_0_1  x_2_1  x_2_2
               /   \    |    |
           x_-1_0 x_-1_1 x_3_1 x_3_2
        """
        G = nx.Graph()
        rows, cols = 4, 3
        
        # Root
        G.add_node('x_1_2')
        
        # Level 1 branches
        level1_nodes = ['x_0_0', 'x_0_1', 'x_2_1', 'x_2_2']
        for node in level1_nodes:
            G.add_node(node)
        
        # Level 2 branches (only some nodes have children)
        level2_nodes = ['x_-1_0', 'x_-1_1', 'x_3_1', 'x_3_2']
        for node in level2_nodes:
            G.add_node(node)
        
        # Root to level 1 factors
        connections_l1 = [('x_1_2', 'x_0_0'), ('x_1_2', 'x_0_1'), 
                          ('x_1_2', 'x_2_1'), ('x_1_2', 'x_2_2')]
        
        for i, (parent, child) in enumerate(connections_l1):
            factor = f'f_root_{i}'
            G.add_node(factor)
            G.add_edge(parent, factor)
            G.add_edge(child, factor)
        
        # Level 1 to level 2 factors
        connections_l2 = [('x_0_1', 'x_-1_0'), ('x_0_1', 'x_-1_1'),
                          ('x_2_1', 'x_3_1'), ('x_2_2', 'x_3_2')]
        
        for i, (parent, child) in enumerate(connections_l2):
            factor = f'f_l1_{i}'
            G.add_node(factor)
            G.add_edge(parent, factor)
            G.add_edge(child, factor)
        
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
        if self.graph_type in ["3x3_grid", "2x3_grid", "4x4_grid"]:
            self._setup_grid_positions(graph)
        elif self.graph_type == "chain":
            self._setup_chain_positions()
        elif self.graph_type == "vertical_chain":
            self._setup_vertical_chain_positions()
        elif self.graph_type == "star":
            self._setup_star_positions()
        elif self.graph_type == "triangle":
            self._setup_triangle_positions()
        elif self.graph_type == "binary_tree":
            self._setup_binary_tree_positions()
        elif self.graph_type == "path_tree":
            self._setup_path_tree_positions()
        elif self.graph_type == "branching_tree":
            self._setup_branching_tree_positions()
        elif self.graph_type == "simple_chain":
            self._setup_simple_chain_positions()
        
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
        for i in range(5):
            self.true_positions[f'x_0_{i}'] = np.array([i, 0])
        
        # Factor positions
        for i in range(4):
            self.positions[f'f_{i}'] = np.array([i + 0.5, 0.3])
    
    def _setup_vertical_chain_positions(self):
        """Setup positions for vertical chain graph."""
        # Variable positions - only set true positions initially
        for i in range(5):
            self.true_positions[f'x_{i}_0'] = np.array([0, 4-i])
        
        # Factor positions
        for i in range(4):
            self.positions[f'f_{i}'] = np.array([0.3, 4-i-0.5])
    
    def _setup_star_positions(self):
        """Setup positions for star graph."""
        # Central node - only set true position initially
        self.true_positions['x_1_1'] = np.array([1, 1])
        
        # Surrounding nodes - only set true positions initially
        true_pos = [(1, 2), (0, 1), (2, 1), (1, 0)]  # top, left, right, bottom
        node_names = ['x_0_1', 'x_1_0', 'x_1_2', 'x_2_1']
        
        for idx, (node, true) in enumerate(zip(node_names, true_pos)):
            self.true_positions[node] = np.array(true)
            self.positions[f'f_{idx}'] = np.array([(1 + true[0])/2, (1 + true[1])/2])
    
    def _setup_triangle_positions(self):
        """Setup positions for triangle graph."""
        # Triangle nodes - only set true positions initially
        node_positions = {
            'x_0_0': np.array([0, 1]),
            'x_0_1': np.array([1, 1]),
            'x_1_0': np.array([0.5, 0])
        }
        
        for node, pos in node_positions.items():
            self.true_positions[node] = pos
        
        # Factor positions
        factor_positions = [
            np.array([0.5, 1.2]),    # f_0: between x_0_0 and x_0_1
            np.array([0.75, 0.5]),   # f_1: between x_0_1 and x_1_0
            np.array([0.25, 0.5])    # f_2: between x_1_0 and x_0_0
        ]
        
        for idx, pos in enumerate(factor_positions):
            self.positions[f'f_{idx}'] = pos
    
    def _setup_binary_tree_positions(self):
        """Setup positions for binary tree graph."""
        # Root
        self.true_positions['x_0_1'] = np.array([2.0, 4.0])
        
        # Level 1
        self.true_positions['x_1_0'] = np.array([1.0, 2.5])
        self.true_positions['x_1_2'] = np.array([3.0, 2.5])
        
        # Level 2 (leaves)
        leaf_x_positions = [0.0, 1.5, 2.5, 4.0]
        for i in range(4):
            self.true_positions[f'x_2_{i}'] = np.array([leaf_x_positions[i], 1.0])
        
        # Factor positions (at midpoints)
        self.positions['f_root_left'] = np.array([1.5, 3.25])
        self.positions['f_root_right'] = np.array([2.5, 3.25])
        self.positions['f_left_0'] = np.array([0.5, 1.75])
        self.positions['f_left_1'] = np.array([1.25, 1.75])
        self.positions['f_right_2'] = np.array([2.75, 1.75])
        self.positions['f_right_3'] = np.array([3.5, 1.75])
    
    def _setup_path_tree_positions(self):
        """Setup positions for path tree graph."""
        # Main path
        for i in range(5):
            self.true_positions[f'x_0_{i}'] = np.array([i * 1.5, 2.0])
        
        # Branches
        for i, pos in enumerate([1, 2, 3]):
            self.true_positions[f'x_1_{pos}'] = np.array([pos * 1.5, 0.5])
        
        # Main path factors
        for i in range(4):
            self.positions[f'f_main_{i}'] = np.array([i * 1.5 + 0.75, 2.3])
        
        # Branch factors
        for pos in [1, 2, 3]:
            self.positions[f'f_branch_{pos}'] = np.array([pos * 1.5, 1.25])
    
    def _setup_branching_tree_positions(self):
        """Setup positions for branching tree graph."""
        # Root
        self.true_positions['x_1_2'] = np.array([2.0, 3.0])
        
        # Level 1
        level1_positions = {
            'x_0_0': np.array([0.5, 2.0]),
            'x_0_1': np.array([1.5, 2.0]),
            'x_2_1': np.array([2.5, 2.0]),
            'x_2_2': np.array([3.5, 2.0])
        }
        self.true_positions.update(level1_positions)
        
        # Level 2
        level2_positions = {
            'x_-1_0': np.array([1.0, 1.0]),
            'x_-1_1': np.array([2.0, 1.0]),
            'x_3_1': np.array([2.5, 1.0]),
            'x_3_2': np.array([3.5, 1.0])
        }
        self.true_positions.update(level2_positions)
        
        # Root to level 1 factors
        root_factor_positions = {
            'f_root_0': np.array([1.25, 2.5]),
            'f_root_1': np.array([1.75, 2.5]),
            'f_root_2': np.array([2.25, 2.5]),
            'f_root_3': np.array([2.75, 2.5])
        }
        self.positions.update(root_factor_positions)
        
        # Level 1 to level 2 factors
        l1_factor_positions = {
            'f_l1_0': np.array([1.25, 1.5]),
            'f_l1_1': np.array([1.75, 1.5]),
            'f_l1_2': np.array([2.5, 1.5]),
            'f_l1_3': np.array([3.5, 1.5])
        }
        self.positions.update(l1_factor_positions)
    
    def _setup_simple_chain_positions(self):
        """Setup positions for simple chain graph."""
        # Variable positions - only set true positions initially
        for i in range(3):
            self.true_positions[f'x_{i}'] = np.array([i * 2.0, 0])
        
        # Factor positions
        self.positions['f_rel_0_1'] = np.array([1.0, 0.5])
        self.positions['f_rel_1_2'] = np.array([3.0, 0.5])
    
    def _initialize_estimated_positions(self, fix_single_node: bool = False):
        """Initialize estimated positions with noise around GT positions."""
        np.random.seed(42)  # For reproducible results
        noise_std = 0.3  # Standard deviation of initialization noise
        
        # Get which nodes should be fixed at GT position
        fixed_nodes = []
        if fix_single_node:
            fixed_nodes = self.get_observation_nodes(fix_single_node)
        
        for var_name, true_pos in self.true_positions.items():
            if var_name in fixed_nodes:
                # Fixed nodes start at ground truth position
                self.positions[var_name] = true_pos.copy()
            else:
                # Add Gaussian noise to true position for initial estimate
                noise = np.random.normal(0, noise_std, size=2)
                self.positions[var_name] = true_pos + noise
    
    def get_observation_nodes(self, fix_single_node: bool = False) -> list:
        """Get observation nodes based on graph type."""
        if fix_single_node:
            # Fix only one source node
            if self.graph_type == "3x3_grid":
                corners = ['x_1_1']  # Center node
            elif self.graph_type == "2x3_grid":
                corners = ['x_0_1']  # Middle node
            elif self.graph_type == "4x4_grid":
                corners = ['x_1_1']  # Center-ish node
            elif self.graph_type == "chain":
                corners = ['x_0_0']  # First node
            elif self.graph_type == "vertical_chain":
                corners = ['x_0_0']  # First node
            elif self.graph_type == "star":
                corners = ['x_1_1']  # Center node
            elif self.graph_type == "triangle":
                corners = ['x_0_0']  # First node
            elif self.graph_type == "binary_tree":
                corners = ['x_0_1']  # Root node
            elif self.graph_type == "path_tree":
                corners = ['x_0_0']  # Start of main path
            elif self.graph_type == "branching_tree":
                corners = ['x_1_2']  # Root node
            elif self.graph_type == "simple_chain":
                corners = ['x_0']  # First node only
            else:
                corners = []
        else:
            # Original multiple observation points
            if self.graph_type == "3x3_grid":
                corners = ['x_0_0', 'x_0_2', 'x_2_0', 'x_2_2']
            elif self.graph_type == "2x3_grid":
                corners = ['x_0_0', 'x_0_2', 'x_1_0', 'x_1_2']
            elif self.graph_type == "4x4_grid":
                corners = ['x_0_0', 'x_0_3', 'x_3_0', 'x_3_3']
            elif self.graph_type == "chain":
                corners = ['x_0_2']  # Middle node only
            elif self.graph_type == "vertical_chain":
                corners = ['x_2_0']  # Middle node only
            elif self.graph_type == "star":
                corners = ['x_1_1']  # Center node
            elif self.graph_type == "triangle":
                corners = ['x_0_0', 'x_1_0']
            elif self.graph_type == "binary_tree":
                corners = ['x_2_0', 'x_2_3']  # Two leaf nodes
            elif self.graph_type == "path_tree":
                corners = ['x_0_0', 'x_0_4']  # Start and end of main path
            elif self.graph_type == "branching_tree":
                corners = ['x_-1_0', 'x_3_2']  # Two leaf nodes
            elif self.graph_type == "simple_chain":
                corners = ['x_0', 'x_2']  # First and last nodes
            else:
                corners = []
        
        return corners