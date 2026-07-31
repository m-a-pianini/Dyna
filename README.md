# DYNA

DYNA is a Python package for studying continuous-time dynamical systems and discrete maps. It combines JAX-compatible vector fields, Diffrax integration, graph-based system composition, Lyapunov-exponent estimation, and geometric analysis of trajectories.

## Purpose

DYNA is designed to support the full workflow of numerical nonlinear-dynamics experiments: define a model, combine interacting subsystems, integrate or iterate its dynamics, and quantify the resulting behavior. It is particularly suited to exploring stability, chaos, attractors, transport, and geometric properties of trajectories in coursework, research prototypes, and reproducible computational experiments.

The package separates model definition from numerical integration. A dynamical system describes states, inputs, parameters, outputs, and update rules, while users choose the appropriate ODE solver or discrete stepping method for each experiment.

## Features

- **Composable dynamical systems:** define continuous-time, discrete, or hybrid systems with named state and input variables.
- **Graph-based coupling:** merge independent systems or connect outputs to inputs, including cyclic feedback networks and nested compositions.
- **JAX compatibility:** use JAX arrays, automatic differentiation, JVPs, vectorization, and JIT compilation for numerical experiments.
- **Diffrax integration:** integrate continuous-time flows with configurable Diffrax solvers, save points, and step-size controllers.
- **Built-in models:** experiment with Lorenz, Duffing, Samelson/Bjerknes-jet, Hamiltonian, and other example vector fields.
- **Discrete maps:** iterate maps such as the standard map and retain complete trajectories including the initial state.
- **Lyapunov analysis:** estimate maximal exponents and flow spectra with finite-difference shadowing or Benettin QR orthogonalization.
- **Geometric analysis:** extract Poincare sections, handle wrapped periodic coordinates, estimate box-counting and correlation dimensions, and compute Kaplan-Yorke dimensions.
- **Parameter and clock utilities:** expose time as a wireable clock input and promote parameters to state variables for fitting or parameter dynamics.
- **Batch experimentation:** create JAX-compatible solver functions that can be vectorized across multiple initial conditions.

## Package structure

```text
dyna/
├── __init__.py       Package metadata
├── dynsys.py         DynamicalSystem objects and composition
├── flows.py          Continuous-time vector fields and plotting helpers
├── maps.py           Discrete maps and map iteration
├── analysis.py       Poincare sections and fractal-dimension analysis
└── lyapunov.py       Lyapunov-exponent algorithms

experiments/
└── oscillator_chain_example.py
```

## Installation

Create and activate an environment, then install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt scipy sympy graph-tool
```

`graph-tool` may need to be installed through the operating system package manager. On Debian or Ubuntu:

```bash
sudo apt install python3-graph-tool
```

The required scientific stack includes NumPy, SciPy, Matplotlib, JAX, Diffrax, SymPy, and graph-tool.

## Dynamical systems

### Atomic systems

`DynamicalSystem` describes a system using named state variables, optional named inputs, parameters, outputs, and a domain (`continuous`, `discrete`, or `hybrid`). Its dynamics function has the signature:

```python
def fn(x, u, params, t):
	...
```

For a continuous system it returns `dx/dt`; for a discrete system it returns the next state.

```python
import jax.numpy as jnp
from dyna.dynsys import DynamicalSystem, VarSpec

def rhs(x, u, params, t):
	position, velocity = x
	return jnp.array([
		velocity,
		-params["k"] * position - params["c"] * velocity,
	])

oscillator = DynamicalSystem(
	name="oscillator",
	state_vars=[VarSpec("position"), VarSpec("velocity")],
	outputs=["position"],
	fn=rhs,
	params={"k": 1.0, "c": 0.2},
	domain="continuous",
)
```

Evaluate a system with `system(x, u, params, t)`. Omitting `u` or `params` uses zero inputs or the system's default parameters.

### Composition

Use `merge` for independent systems:

```python
from dyna.dynsys import merge

combined = merge(system_a, system_b, name="combined")
```

Use `connect` to wire named outputs to named inputs. Edges are tuples of the form `(source_system, source_output, destination_system, destination_input)`:

```python
from dyna.dynsys import connect

coupled = connect(
	[oscillator_a, oscillator_b],
	[
		("oscillator_a", "position", "oscillator_b", "coupling_in"),
		("oscillator_b", "position", "oscillator_a", "coupling_in"),
	],
	name="coupled",
)
```

Feedback loops are supported. Outputs are read synchronously from the current global state, so composition does not introduce algebraic loops.

Convenience operators are available:

```python
merged = system_a | system_b
cascade = system_a >> system_b
```

`series(a, b)` connects equally sized outputs and inputs with matching names.

### Clocks and parameters

`make_clock` creates a wireable clock system for time-dependent compositions. `autonomize` exposes a system's implicit time argument as an input. `promote_params` converts selected parameters into constant state variables, which is useful for parameter fitting or slow parameter dynamics.

## Built-in flows

`dyna.flows` contains the following example vector fields and helpers:

- `hamiltonian_flow`
- `lorenz_system`
- `duffing`
- `samelson_flow`
- `ichikievich`
- `trajectory_plot` and `stream_plot`

Example Diffrax integration:

```python
import diffrax as dfx
import jax.numpy as jnp
from dyna.flows import lorenz_system

params = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
z0 = jnp.array([1.0, 1.0, 1.0])
term = dfx.ODETerm(lambda t, z, args: lorenz_system(t, z, args))

solution = dfx.diffeqsolve(
	term,
	dfx.Dopri5(),
	t0=0.0,
	t1=50.0,
	dt0=0.01,
	y0=z0,
	args=params,
	saveat=dfx.SaveAt(ts=jnp.linspace(0.0, 50.0, 5000)),
)
```

## Discrete maps

`dyna.maps` provides map iteration and the standard map:

```python
import numpy as np
from dyna.maps import iterate_map, standard_map

trajectory = iterate_map(
	lambda x: standard_map(x, k=1.2),
	np.array([0.1, 0.2]),
	N=1000,
)
```

The returned array has shape `(N + 1, dimension)` and includes the initial state.

## Lyapunov analysis

`dyna.lyapunov` provides:

- `mLCE_map`: maximal exponent for a discrete map;
- `mLCE_flow`: finite-difference estimate for a continuous flow;
- `flow_lyapunov_spectrum`: QR-based spectrum estimation;
- `fast_flow_lyapunov_spectrum`: lower-memory spectrum calculation;
- batch solver factories for multiple initial conditions.

```python
from dyna.lyapunov import mLCE_map

exponent = mLCE_map(
	lambda x: standard_map(x, k=1.2),
	np.array([0.1, 0.1]),
	N=10000,
)
```

The flow spectrum functions use JAX automatic differentiation or JVPs and Diffrax integration. They implement the Benettin algorithm with QR orthogonalization.

## Trajectory analysis

`dyna.analysis` contains:

- `poincare_sos`: extract samples near a section;
- `plot_wrapped`: plot periodic coordinates without connecting wrap discontinuities;
- `count_boxes` and `boxcount_dimension`: estimate box-counting dimension;
- `correlation_dimension`: estimate correlation dimension;
- `kaplan_yorke_dim`: compute the Kaplan-Yorke dimension;
- `find_stationary`: search for candidate stationary points;
- `find_linear_region`: identify a linear fit region.

Example:

```python
from dyna.analysis import boxcount_dimension

dimension, sizes, counts, start, end = boxcount_dimension(
	np.asarray(solution.ys),
)
```

## Running the example

From the repository root:

```bash
python experiments/oscillator_chain_example.py
```

The example builds a five-oscillator cyclic network, evaluates its composite dynamics, and demonstrates JAX compilation with subsystem-specific parameters.

## Current limitations

The following parts are experimental or incomplete:

- The Hénon map is not implemented.
- `flow_mLCE` and `map_lyapunov_spectrum` are placeholders.
- `correlation_dimension` is marked for future correction.
- Some example code uses version-dependent NumPy or JAX APIs.
- Composite parameter dictionaries must preserve the nesting expected by `CompositeSystem`; use `flatten_params` and `unflatten_params` when needed.

Simulation and integration are intentionally separate from `DynamicalSystem`: users choose the appropriate solver and stepping strategy for each domain.

## Development notes

The package relies on JAX arrays and transformations. Avoid unnecessary NumPy conversions inside differentiated or JIT-compiled functions. For reproducible numerical experiments, set random seeds for Lyapunov estimates, enable JAX 64-bit arithmetic when needed, discard transients before measuring attractor properties, and inspect the fitted scale range when estimating fractal dimensions.
