#!/usr/bin/env python3
"""
Gaussian Belief Propagation Animation with Dynamic Edges
Refactored implementation using modular design
"""

import sys
from src.visualization import DynamicEdgeBPAnimation


def main():
    """Main function with graph type selection."""
    # Check for default mode flag
    if len(sys.argv) > 1 and sys.argv[1] == '-d':
        # Default mode: 3x3_grid with anchor
        print("=== Default Mode: 3x3 Grid with Anchor ===")
        demo = DynamicEdgeBPAnimation(
            graph_type="3x3_grid", 
            fix_single_node=True,
            relative_weight=1.0,
            smoothness_weight=0.1,
            anchor_weight=10000.0
        )
        demo.run_animation()
        return
    
    print("=== Gaussian BP Animation with Multiple Graph Types ===")
    print("Available graph types:")
    print("1. 3x3_grid - 3x3 grid graph (default)")
    print("2. 2x3_grid - 2x3 grid graph")
    print("3. 4x4_grid - 4x4 grid graph")
    print("4. chain - Linear chain graph")
    print("5. star - Star graph")
    print("6. triangle - Triangle graph")
    print("7. binary_tree - Binary tree (demonstrates one-way message flow)")
    print("8. path_tree - Path with branches (tree structure)")
    print("9. branching_tree - Multi-branch tree")
    print("10. simple_chain - Simple 3-node 2-edge chain (for debugging)")
    print("\nTip: Use -d flag for quick default execution (simple_chain with anchor)")
    
    try:
        choice = input("\nEnter choice (1-10) or press Enter for default: ").strip()
        
        graph_types = {
            "1": "3x3_grid",
            "2": "2x3_grid", 
            "3": "4x4_grid",
            "4": "chain",
            "5": "star",
            "6": "triangle",
            "7": "binary_tree",
            "8": "path_tree",
            "9": "branching_tree",
            "10": "simple_chain"
        }
        
        graph_type = graph_types.get(choice, "chain")
        
        # Ask about fixing single node
        fix_choice = input("\nFix only one source node? (y/n, default=n): ").strip().lower()
        fix_single_node = fix_choice in ['y', 'yes']
        
        # Ask about factor weights
        weight_choice = input("\nCustomize factor weights? (y/n, default=n): ").strip().lower()
        
        relative_weight = 1.0
        smoothness_weight = 0.1
        anchor_weight = 10000.0
        
        if weight_choice in ['y', 'yes']:
            try:
                relative_input = input(f"Relative position factor weight (default={relative_weight}): ").strip()
                if relative_input:
                    relative_weight = float(relative_input)
                
                smoothness_input = input(f"Smoothness factor weight (default={smoothness_weight}): ").strip()
                if smoothness_input:
                    smoothness_weight = float(smoothness_input)
                
                if fix_single_node:
                    anchor_input = input(f"Anchor factor weight (default={anchor_weight}): ").strip()
                    if anchor_input:
                        anchor_weight = float(anchor_input)
            except ValueError:
                print("Invalid input, using default weights")
        
        print(f"\nStarting animation with {graph_type}")
        print(f"• Fix single node: {fix_single_node}")
        print(f"• Relative weight: {relative_weight}")
        print(f"• Smoothness weight: {smoothness_weight}")
        if fix_single_node:
            print(f"• Anchor weight: {anchor_weight}")
        
        demo = DynamicEdgeBPAnimation(
            graph_type=graph_type, 
            fix_single_node=fix_single_node,
            relative_weight=relative_weight,
            smoothness_weight=smoothness_weight,
            anchor_weight=anchor_weight
        )
        demo.run_animation()
        
    except KeyboardInterrupt:
        print("\nAnimation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to default
        print("Using default 3x3 grid...")
        demo = DynamicEdgeBPAnimation()
        demo.run_animation()

def run_specific_graph(graph_type="3x3_grid", fix_single_node=False, 
                      relative_weight=1.0, smoothness_weight=0.1, anchor_weight=10000.0):
    """Run animation with specific graph type and weights."""
    demo = DynamicEdgeBPAnimation(
        graph_type=graph_type, 
        fix_single_node=fix_single_node,
        relative_weight=relative_weight,
        smoothness_weight=smoothness_weight,
        anchor_weight=anchor_weight
    )
    demo.run_animation()


if __name__ == "__main__":
    main()