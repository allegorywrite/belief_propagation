# Gaussian Belief Propagation Implementation

This repository implements Gaussian Belief Propagation (GBP) for multi-robot localization based on the paper "A Robot Web for Distributed Many-Device Localisation" (2202.03314v2).

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install numpy matplotlib networkx
```

## Usage

### Quick Default Mode
Run with default settings (3x3_grid with anchor):
```bash
python -m src.gaussian_bp_animation -d
```

### Interactive Mode
Run the animation with graph type selection:
```bash
python -m src.gaussian_bp_animation
```

Available graph types:
1. 3x3_grid - 3x3 grid graph (default)
2. 2x3_grid - 2x3 grid graph
3. 4x4_grid - 4x4 grid graph
4. chain - Linear chain graph
5. star - Star graph
6. triangle - Triangle graph
7. binary_tree - Binary tree (demonstrates one-way message flow)
8. path_tree - Path with branches (tree structure)
9. branching_tree - Multi-branch tree
10. **simple_chain - Simple 3-node 2-edge chain (for debugging)**

### Direct Execution Examples

#### Simple Chain with Anchor Node (Recommended for debugging)
```bash
echo -e "10\ny" | python -m src.gaussian_bp_animation
```
- **Result**: Perfect convergence in 3 iterations (Total error: 0.000)
- **Use case**: Debugging message propagation algorithm

#### Simple Chain without Anchor
```bash
echo -e "10\nn" | python -m src.gaussian_bp_animation
```
- **Result**: Converges but with remaining error due to translation freedom
- **Use case**: Understanding the need for reference frames

#### 3x3 Grid with Single Anchor
```bash
echo -e "1\ny" | python -m src.gaussian_bp_animation
```

#### Custom Factor Weights
```bash
# Test with dominant smoothness factor
echo -e "10\ny\ny\n0.1\n5.0\n10000.0" | python -m src.gaussian_bp_animation

# Test with no relative position constraints
echo -e "10\ny\ny\n0.0\n0.1\n10000.0" | python -m src.gaussian_bp_animation
```

## Implementation Features

### Factor Types
- **Relative Position Factors**: Linear factors constraining relative positions between adjacent nodes
  - Measurement model: `z = x2 - x1` (relative position)
  - Jacobian: `[-I, I]` (2x4 matrix for 2D positions)

- **Anchor Factors**: High-precision prior factors for reference frame
  - Only used when `fix_single_node=True`
  - Provides absolute position constraint for one node

### Graph Structures
- **Simple Chain**: 3 nodes with 2 relative position factors (ideal for debugging)
- **Grid Graphs**: Regular grid structures with horizontal and vertical factors
- **Tree Structures**: Hierarchical graphs demonstrating directional message flow

### Convergence Analysis
The implementation correctly demonstrates:
- **With anchor**: Complete convergence to ground truth
- **Without anchor**: Convergence of relative positions with translation freedom
- **Message propagation**: Proper variable-to-factor and factor-to-variable message updates

### Visualization Features
- **Two-window display**: 
  - Main window: Graph visualization with BP animation
  - Convergence window: Real-time error plots showing per-edge errors and total error
- **Dynamic error tracking**: Monitor convergence progress with separate plots for each edge factor
- **Factor weight adjustment**: Configurable weights for relative position, smoothness, and anchor factors

## Key Files
- `src/gaussian_bp.py` - Core GBP algorithm implementation
- `src/gaussian_message.py` - Gaussian message data structures
- `src/graph_utils.py` - Graph creation and position management
- `src/animation.py` - Visualization and animation with convergence plots
- `src/gaussian_bp_animation.py` - Main entry point with CLI interface

## Advanced Usage

### Factor Weight Examples
```bash
# Default balanced weights
python -m src.gaussian_bp_animation -d

# High smoothness (resists change)
echo -e "10\ny\ny\n1.0\n5.0\n10000.0" | python -m src.gaussian_bp_animation

# No smoothness (pure relative position)
echo -e "10\ny\ny\n1.0\n0.0\n10000.0" | python -m src.gaussian_bp_animation

# Weak anchor (less rigid fixing)
echo -e "10\ny\ny\n1.0\n0.1\n100.0" | python -m src.gaussian_bp_animation
```

### Command Line Options
- `-d`: Default mode (3x3_grid with anchor, standard weights)
- Interactive mode: Full customization of graph type, anchor settings, and factor weights

## Theory Reference
Based on the Gaussian Belief Propagation formulation in:
- Paper: "A Robot Web for Distributed Many-Device Localisation" (2202.03314v2)
- Section III: Gaussian Belief Propagation fundamentals
- Equations (11-12): Linear factor potential computation

### Factor Types Implemented
1. **Relative Position Factors**: Enforce true relative positions between adjacent nodes
2. **Smoothness Factors**: Encourage maintenance of current estimated relative positions  
3. **Anchor Factors**: Fix specific nodes to ground truth positions for reference frame
