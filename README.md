# Gaussian Belief Propagation Implementation

## Project Structure

```
belief_propagation/
├── main.py                           # Main entry point
├── src/
│   ├── gaussian_bp.py                # Core BP algorithm
│   ├── gaussian_message.py           # Gaussian message data structure
│   ├── utils/                        # Algorithm utilities
│   │   ├── linear_algebra_utils.py   # Schur complement & matrix operations
│   │   ├── factor_utils.py           # Factor creation utilities
│   │   ├── graph_utils.py            # Graph topology creation
│   │   └── bp_problem_setup.py       # BP problem configuration
│   └── visualization/                # Visualization & animation
│       └── bp_visualization.py       # Interactive BP animation
└── anydocgen/                        # Documentation & references
    └── refs/build/                   # Paper references
```

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

### Quick Start (Default Mode)
Run with default settings (3x3 grid with single anchor):
```bash
python main.py -d
```

### Interactive Mode
Select graph type and parameters interactively:
```bash
python main.py
```