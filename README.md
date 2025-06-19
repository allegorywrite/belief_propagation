# Gaussian Belief Propagation Implementation

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