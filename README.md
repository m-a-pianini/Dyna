# Dynamical Systems Analysis

This repository studies nonlinear dynamical systems using Lyapunov exponent estimation and fractal dimension analysis. It combines numerical flow definitions, trajectory visualization, and tools for quantifying chaos using both map and continuous-time systems.

## Repository Overview

- `src/lyapunov.py` - Core utilities for computing Lyapunov exponents, box-counting fractal dimension, Poincaré section extraction, and other diagnostic functions.
- `src/flows.py` - Example dynamical systems and flow definitions, including the Samelson/Bjerknes jet model, Lorenz system, Duffing oscillator, and Hamiltonian flow wrapper.
- `experiments/` - Notebooks and scripts for exploratory analysis, visualization, and parameter sweeps.
- `figs/`, `figs_backflow/`, `figs_interpars/` - Output directories for generated figures and plots.

## Key Concepts

- **Maximal Lyapunov Exponent (mLCE)**: Measures the average exponential rate of separation of nearby trajectories and is used to identify chaotic behavior.
- **Lyapunov Spectrum**: A full set of Lyapunov exponents that characterize expansion and contraction rates in different directions of phase space.
- **Fractal Dimension Estimates**: Uses box-counting and correlation-based approaches to estimate geometric complexity of attractors.
- **Poincaré Section / Wrapped plots**: Visualization tools for periodic flows and angle-like variables.

## Dependencies

The main Python dependencies are listed in `requirements.txt`:

- `numpy`
- `matplotlib`
- `jax`
- `diffrax`

Additional packages used in the repository:

- `scipy`
- `sympy`

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scipy sympy
```

If using conda:

```bash
conda create -n dynsys python=3.11
conda activate dynsys
pip install -r requirements.txt
pip install scipy sympy
```

## Usage

### Running experiments

Open one of the notebooks in `experiments/` to explore specific models and analysis workflows, such as:

- `experiments/samelson.ipynb` — Samelson flow, streamplots, Lyapunov exponent experiments, and parameter scans.
- `experiments/duffing.ipynb` — Duffing oscillator examples.
- `experiments/lorenz.ipynb` — Lorenz system exploration.

### Importing the library

Use the core modules in your own scripts or notebooks:

```python
from src.flows import samelson_flow, duffing, lorenz_system
from src.lyapunov import plot_wrapped, boxcount_dimension, mLCE_flow
```

### Core functions

- `mLCE_map` — Estimate the maximal Lyapunov exponent for discrete maps.
- `mLCE_flow` — Estimate the maximal Lyapunov exponent for continuous flows using shadowing with RK4.
- `flow_lyapunov_spectrum` / `fast_flow_lyapunov_spectrum` — Compute Lyapunov spectra for flows using Benettin/QR methods.
- `boxcount_dimension` — Estimate fractal dimension from trajectory data.
- `kaplan_yorke_dim` — Compute the Kaplan–Yorke dimension from Lyapunov exponents.
- `plot_wrapped` — Plot periodic variables while handling discontinuities.

## Recommended Workflow

1. Define the dynamical system in `src/flows.py` or create a new flow/map function.
2. Use Diffrax solvers and `dfx.ODETerm` to integrate trajectories.
3. Apply Lyapunov and fractal dimension utilities from `src/lyapunov.py`.
4. Visualize results in notebooks or save figures to `figs/`.

## Notes

- Many exploratory calculations in `experiments/` rely on JAX and Diffrax for efficient numerical integration and automatic differentiation.
- Some functions in `src/lyapunov.py` are still under active development and may include incomplete TODOs for extended spectrum computation and batch solvers.

## Contribution

This repository is designed for research and coursework on complex systems. Contributions may include:

- implementing complete Lyapunov spectrum support
- adding new flows and maps
- improving fractal dimension estimators
- creating additional experiment notebooks

---

If you want to run a specific experiment or extend a model, start with `experiments/samelson.ipynb` and `src/lyapunov.py`.
