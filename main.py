#!/usr/bin/env python3
"""
Gaussian Belief Propagation Animation with Dynamic Edges
Refactored implementation using modular design
"""

import sys
import argparse
from src.visualization import DynamicEdgeBPAnimation


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Gaussian Belief Propagation Animation')
    
    parser.add_argument('-g', '--graph-type', type=str, 
                       choices=['n_chain', 'nxn_grid', 'particle_terminal_chain', 'particle_terminal_3x3_grid'],
                       default='n_chain',
                       help='Graph type to use for belief propagation')
    
    parser.add_argument('-f', '--fix-first-node', action='store_true',
                       help='Fix the first node at ground truth position')
    
    parser.add_argument('--no-viz', action='store_true',
                       help='Run without visualization')
    
    parser.add_argument('-d', '--default', action='store_true',
                       help='Run default mode (3x3_grid with anchor)')
    
    parser.add_argument('--relative-weight', type=float, default=100.0,
                       help='Weight for relative position factors')
    
    parser.add_argument('--smoothness-weight', type=float, default=0.0,
                       help='Weight for smoothness factors')
    
    parser.add_argument('--anchor-weight', type=float, default=10000.0,
                       help='Weight for anchor factors (when fixing nodes)')
    
    parser.add_argument('--particle-nodes', type=str, nargs='*',
                       help='Nodes to convert to particle nodes (e.g., x_0 x_2_2)')
    
    parser.add_argument('--no-message-animation', action='store_true',
                       help='Suppress message propagation animation')
    
    parser.add_argument('--no-variance-display', action='store_true',
                       help='Suppress variance/uncertainty ellipse display')
    
    parser.add_argument('--no-error-window', action='store_true',
                       help='Suppress error convergence window display')
    
    parser.add_argument('-n', '--size', type=int, default=None,
                       help='Size parameter: grid size for nxn_grid (default: 10), chain length for n_chain (default: 5)')

    args = parser.parse_args()
    
    # Handle default mode
    if args.default:
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
    
    # Run with specified parameters
    graph_type = args.graph_type
    fix_single_node = args.fix_first_node
    
    # Parse particle node configuration
    particle_config = {}
    if args.particle_nodes:
        for node_id in args.particle_nodes:
            particle_config[node_id] = {
                'num_particles': 50,
                'dim': 2,
                'noise_scale': 1.0  # Origin-centered initialization
            }
    
    # Set size parameter based on graph type
    if args.size is None:
        # Set defaults based on graph type
        if args.graph_type == 'nxn_grid':
            size = 10
        elif args.graph_type == 'n_chain':
            size = 5
        else:
            size = None
    else:
        size = args.size
    
    print(f"=== Running {graph_type} (visualization: {'off' if args.no_viz else 'on'}) ===")
    print(f"• Fix single node: {fix_single_node}")
    print(f"• Relative weight: {args.relative_weight}")
    print(f"• Smoothness weight: {args.smoothness_weight}")
    if fix_single_node:
        print(f"• Anchor weight: {args.anchor_weight}")
    if particle_config:
        print(f"• Particle nodes: {list(particle_config.keys())}")
    
    demo = DynamicEdgeBPAnimation(
        graph_type=graph_type,
        fix_single_node=fix_single_node,
        relative_weight=args.relative_weight,
        smoothness_weight=args.smoothness_weight,
        anchor_weight=args.anchor_weight,
        particle_nodes=particle_config,
        no_message_animation=args.no_message_animation,
        no_variance_display=args.no_variance_display,
        no_error_window=args.no_error_window,
        grid_size=size
    )
    
    if args.no_viz:
        demo.run_without_animation()
    else:
        demo.run_animation()
    
    # Keep the original interactive mode if no arguments provided
    if len(sys.argv) == 1:
        run_interactive_mode()


def run_interactive_mode():
    """Run the original interactive mode for backward compatibility."""
    print("=== Gaussian BP Animation with Multiple Graph Types ===")
    print("Available graph types:")
    print("1. n_chain - N-node linear chain graph (default 5 nodes)")
    print("2. nxn_grid - NxN grid graph (default 10x10)")
    print("3. particle_terminal_chain - 5-node chain with particle terminal node")
    print("4. particle_terminal_3x3_grid - 3x3 grid with particle terminal node")
    print("\nTip: Use -d flag for quick default execution (3x3_grid with anchor)")
    
    try:
        choice = input("\nEnter choice (1-6) or press Enter for default: ").strip()
        
        graph_types = {
            "1": "n_chain",
            "2": "nxn_grid",
            "3": "particle_terminal_chain",
            "4": "particle_terminal_3x3_grid"
        }
        
        graph_type = graph_types.get(choice, "n_chain")
        
        # Ask about fixing single node
        fix_choice = input("\nFix only one source node? (y/n, default=n): ").strip().lower()
        fix_single_node = fix_choice in ['y', 'yes']
        
        # Ask about factor weights
        weight_choice = input("\nCustomize factor weights? (y/n, default=n): ").strip().lower()
        
        relative_weight = 100.0
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