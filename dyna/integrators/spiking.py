from __future__ import annotations
from typing import Any, Optional, Sequence, Union

import jax.numpy as jnp

from dyna.dynsys import DynamicalSystem

from .base import CondFn, JumpFn, HybridSolution, _as_step, integrate_hybrid, integrate_hybrid_jit, integrate_traj_hybrid_jit

# --------------------------------------------------------------------------
# Common recipe: spiking-neuron voltage threshold/reset + discrete synapses
# --------------------------------------------------------------------------

def make_threshold_condition(
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
) -> CondFn:
    """Return the first threshold condition across the selected voltages."""
    voltage_indices_arr = jnp.asarray(voltage_indices)
    threshold_arr = jnp.broadcast_to(jnp.asarray(threshold, dtype=jnp.float64), voltage_indices_arr.shape)

    def cond_fn(t, y, args, **kwargs):
        v = y[voltage_indices_arr]
        return jnp.min(threshold_arr - v)

    return cond_fn


def make_spike_jump(
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
    v_reset: Union[float, Sequence[float]],
    synapse_step_fn: Optional[JumpFn] = None,
) -> JumpFn:
    voltage_indices_arr = jnp.asarray(voltage_indices)
    threshold_arr = jnp.broadcast_to(jnp.asarray(threshold, dtype=jnp.float64), voltage_indices_arr.shape)
    v_reset_arr = jnp.broadcast_to(jnp.asarray(v_reset, dtype=jnp.float64), voltage_indices_arr.shape)

    def jump_fn(ts, y, args):
        v = y[voltage_indices_arr]
        margins = threshold_arr - v
        spiked = jnp.logical_or(
            margins <= jnp.asarray(1e-5, dtype=v.dtype),
            margins == jnp.min(margins),
        )
        y = y.at[voltage_indices_arr].set(jnp.where(spiked, v_reset_arr, v))
        if synapse_step_fn is not None:
            y = synapse_step_fn(ts, y, args)
        return y

    return jump_fn


def spiking_integrator(
    system: DynamicalSystem,
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
    v_reset: Union[float, Sequence[float]],
    y0: jnp.ndarray,
    args: Any,
    t0: float,
    t1: float,
    dt0: float,
    update_synapses: bool = True,
    **kwargs: Any,
) -> HybridSolution:
    """
    The common use case: integrate `system` (typically a CompositeSystem
    wiring together continuous spiking-neuron voltage dynamics with a
    discrete synapse-weight update rule -- e.g. STDP -- driven by spike
    times) with exact, event-driven spike detection and reset.

    voltage_indices : indices into `system`'s GLOBAL state vector of each
                      neuron's own membrane-voltage variable, e.g. via
                      `system.state_slice("neuron0.v")` on a composite.
    threshold, v_reset : a scalar (shared by every neuron) or one value per
                      entry of `voltage_indices`.
    update_synapses : if True (default), `system.step` runs in the SAME jump
                      as the voltage reset -- this is where any discretely-
                      updated synapse weights (composed as ordinary discrete
                      subsystems, see CompositeSystem) actually see the spike
                      and update. Set False to reset voltages only.
    **kwargs        : forwarded to `integrate_hybrid` (solver, root_finder,
                      saveat, max_events, diffeqsolve_kwargs).

    This is a thin, fully-inspectable composition of `integrate_hybrid` +
    `make_threshold_condition` + `make_spike_jump` -- read those (or this
    function's body) as a template for a different recipe on a hybrid system
    that doesn't fit this one.
    """
    cond_fn = make_threshold_condition(voltage_indices, threshold)
    synapse_step_fn = _as_step(system) if update_synapses else None
    jump_fn = make_spike_jump(voltage_indices, threshold, v_reset, synapse_step_fn=synapse_step_fn)
    return integrate_hybrid(system, cond_fn, jump_fn, y0, args, t0, t1, dt0, **kwargs)


def spiking_integrator_jit(
    system: DynamicalSystem,
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
    v_reset: Union[float, Sequence[float]],
    y0: jnp.ndarray,
    args: Any,
    t0: float,
    t1: float,
    dt0: float,
    max_events: int,
    update_synapses: bool = True,
    **kwargs: Any,
) -> tuple:
    """
    jit/grad/vmap-compatible counterpart to `spiking_integrator` -- same
    recipe (threshold spike + reset + in-jump synapse `.step`), built on
    `integrate_hybrid_jit`. See that function's docstring for the returned
    tuple shape and the `max_events`/no-dense-trajectory trade-offs.
    """
    cond_fn = make_threshold_condition(voltage_indices, threshold)
    synapse_step_fn = _as_step(system) if update_synapses else None
    jump_fn = make_spike_jump(voltage_indices, threshold, v_reset, synapse_step_fn=synapse_step_fn)
    return integrate_hybrid_jit(system, cond_fn, jump_fn, y0, args, t0, t1, dt0, max_events, **kwargs)


def spiking_integrator_jit_trajectory(
    system: DynamicalSystem,
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
    v_reset: Union[float, Sequence[float]],
    y0: jnp.ndarray,
    args: Any,
    t0: float,
    t1: float,
    dt0: float,
    n_intervals: int,
    update_synapses: bool = True,
    **kwargs: Any,
) -> tuple:
    """
    jit/grad/vmap-compatible counterpart to `spiking_integrator` -- same
    recipe (threshold spike + reset + in-jump synapse `.step`), built on
    `integrate_hybrid_jit`. See that function's docstring for the returned
    tuple shape and the `max_events`/no-dense-trajectory trade-offs.
    """
    cond_fn = make_threshold_condition(voltage_indices, threshold)
    synapse_step_fn = _as_step(system) if update_synapses else None
    jump_fn = make_spike_jump(voltage_indices, threshold, v_reset, synapse_step_fn=synapse_step_fn)
    return integrate_traj_hybrid_jit(system, cond_fn, jump_fn, y0, args, t0, t1, n_intervals, **kwargs)
