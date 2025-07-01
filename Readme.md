# Geometry Optimizer

This project contains a set of utilities for running piston geometry optimization using evolutionary and Bayesian algorithms.

The main entry point is `main.py` which exposes a GUI as well as CLI helpers. It integrates:

- **NSGA‑III algorithms** implemented in `nsga3_algorithm.py` for multi‑objective optimization.
- **Bayesian optimization** logic in `bayesian_optimization.py` for expensive evaluations.
- **Constraint management** provided by `constraints.py` to enforce custom parameter constraints.
- **CPU affinity utilities** (`cpu_affinity.py`) that ensure simulations avoid using the first CPU core.
- **Piston simulation wrapper** in `piston_optimizer.py` that manages execution of `fsti_gap.exe`.
- **Post-optimization plotting** provided by `plot_parameter_evolution.py` to visualize parameter trends.

A small `tests/` folder is provided with a pytest test for the Pareto domination sort implementation.

## Requirements

The scripts require Python 3.7 or later. Some features rely on additional packages such as:

- `numpy`, `pandas`, `psutil`
- For Bayesian optimization: `scipy` and `scikit-learn`
- `tkinter` for the GUI

Install dependencies with `pip install -r requirements.txt` or manually install the packages listed above.

## Usage

```bash
python main.py             # start the GUI
python main.py --cli       # use the command‑line interface
python main.py --config my_config.json   # run with a saved configuration
```

Example configuration files can be generated via the `--save-default` flag or the GUI. The optimization scripts assume the presence of `fsti_gap.exe` and an `input/` folder inside the base directory supplied in the configuration.

## Testing

Unit tests use `pytest`. From the repository root run:

```bash
pytest -q
```

The test suite currently only covers the domination sort routine but can be extended to verify other components.

## Repository Structure

```
Readme.md                 gui_interface.py     tests/
bayesian_optimization.py  main.py              visualization.py
plot_parameter_evolution.py
constraints.py            nsga3_algorithm.py
cpu_affinity.py           piston_optimizer.py
```

Each module contains detailed docstrings describing its purpose and behaviour.

