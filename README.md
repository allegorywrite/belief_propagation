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

### 基本実行
```bash
source venv/bin/activate
python main.py -g <graph_type> [options]
```

### CLI引数
- `-g, --graph-type`: グラフタイプ (3x3_grid, chain, particle_terminal_chain等)
- `-f, --fix-first-node`: 最初のノードを固定
- `--no-viz`: 可視化なしで実行
- `--particle-nodes`: パーティクルノードに変換するノード名 (例: x_0 x_2_2)

### 実行例
```bash
# 基本実行
python main.py -g chain --no-viz

# ノード固定
python main.py -g 3x3_grid -f --no-viz

# パーティクルノード追加
python main.py -g chain --particle-nodes x_0_4 --no-viz

# 複合
python main.py -g 3x3_grid -f --particle-nodes x_2_2 --no-viz

# インタラクティブモード
python main.py
```