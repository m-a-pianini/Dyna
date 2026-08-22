"""
integrators.py
===============

Event-driven integration for hybrid (continuous + discrete) DynamicalSystems
and CompositeSystems, on top of `diffrax`.

Layered design
--------------
1. `integrate_hybrid(...)` -- the generic primitive. It knows NOTHING about
   spiking neurons, synapses, or any specific system: you give it
     - `system`  : a DynamicalSystem/CompositeSystem (its `.flow` is used)
                   or a bare diffrax-style vector field `(t, y, args) -> dy`,
     - `cond_fn` : `(t, y, args) -> scalar`, root-found via `diffrax.Event`
                   -- the event fires at the exact time this crosses zero,
     - `jump_fn` : `(t, y, args) -> y_new` -- applied exactly at that time,
                   free to touch ANY part of the state (continuous or
                   discrete slots alike; e.g. reset a neuron's voltage AND
                   update synapse weights in the same jump),
   and it integrates the flow, root-finds the event, applies the jump,
   and resumes -- repeating until `t1` or `max_events`.

2. `make_threshold_condition` / `make_spike_jump` -- small, inspectable
   building blocks for the common "voltage crosses a threshold" event and
   "reset the spiking neuron(s), then let the composite apply whatever
   discrete update is wired in" jump.

3. `spiking_integrator(...)` -- the common use case (spiking-neuron voltage
   dynamics + discretely-updated synapse weights), built ENTIRELY out of
   layers 1-2. If your hybrid system doesn't fit this recipe (different
   reset rule, multiple thresholds, a non-spiking event...), read this
   function as a template and call `integrate_hybrid` directly with your
   own `cond_fn`/`jump_fn` instead -- that's the whole point of keeping it
   layered rather than one big configurable function.

Why a Python loop over `diffeqsolve` calls, rather than one call
------------------------------------------------------------------
A single `diffeqsolve(..., event=...)` call integrates up to (and stops
exactly at) ONE event. Repeated events -- e.g. a neuron spiking many times
over a simulation -- are handled by calling `diffeqsolve` again from the
post-jump state, in an ordinary (eager) Python loop; each individual
`diffeqsolve` call is itself efficient/jittable, but the number of spikes is
data-dependent and unknown ahead of time, so the outer loop over calls is
not itself jitted. This is the standard pattern for repeated diffrax events.

Diffrax version note
---------------------
This uses `diffrax.Event` (root-finding via an `optimistix` root finder),
available in reasonably recent diffrax. Exactly how a solve reports "an
event happened before t1" varies a little by version; see the comment next
to `event_fired` below if you hit an AttributeError there on your version --
the `sol.ts[-1] < t1` fallback should hold regardless.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import optimistix as optx

from ..dynsys import DynamicalSystem

__all__ = [
    "HybridSolution",
    "integrate_hybrid",
    "make_threshold_condition",
    "make_spike_jump",
    "spiking_integrator",
]

FlowFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]   # (t, y, args) -> dy
CondFn = Callable[..., Any]                               # (t, y, args, **kwargs) -> scalar
JumpFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]   # (t, y, args) -> y_new


class HybridSolution(NamedTuple):
    """A stitched-together trajectory across however many events fired.
    `event_times` records the exact (root-found) time of every jump --
    often the actual quantity of interest for spiking-neuron work (e.g. for
    computing STDP updates or spike-train statistics after the fact)."""
    ts: jnp.ndarray
    ys: jnp.ndarray
    event_times: jnp.ndarray


def _as_flow(system: Union[DynamicalSystem, FlowFn]) -> FlowFn:
    """Accept either a raw diffrax-style vector field, or any
    DynamicalSystem/CompositeSystem -- using its `.flow` (zero on any
    discrete part, see dynsys.py). Any *external* driving input should be
    wired in via composition (e.g. `make_clock` + `connect`) rather than
    threaded through here, to keep this integrator agnostic to what `system`
    actually is."""
    if isinstance(system, DynamicalSystem):
        u0 = jnp.zeros((system.input_size,))
        return lambda t, y, args: system.flow(y, u0, args, t)
    return system


def _as_step(system: Union[DynamicalSystem, JumpFn]) -> JumpFn:
    """Same idea as `_as_flow`, but for `.step` -- used by `spiking_integrator`
    to apply a composite's discrete part (e.g. synapse updates) inside a jump."""
    if isinstance(system, DynamicalSystem):
        u0 = jnp.zeros((system.input_size,))
        return lambda t, y, args: system.step(y, u0, args, t)
    return system


def integrate_hybrid(
    system: Union[DynamicalSystem, FlowFn],
    cond_fn: CondFn,
    jump_fn: JumpFn,
    y0: jnp.ndarray,
    args: Any,
    t0: float,
    t1: float,
    dt0: float,
    solver: Optional["dfx.AbstractSolver"] = None,
    root_finder: Optional["optx.AbstractRootFinder"] = None,
    saveat: Optional["dfx.SaveAt"] = None,
    max_events: int = 10_000,
    diffeqsolve_kwargs: Optional[dict] = None,
) -> HybridSolution:
    """
    Generic event-driven hybrid integrator (see module docstring). Repeatedly:
      1. integrates `system`'s flow from the current (t, y) towards `t1`,
      2. stops early -- at the diffrax/optimistix root-found exact time --
         the moment `cond_fn(t, y, args)` crosses zero,
      3. applies `jump_fn` there to get the post-event state,
      4. resumes from (event_time, post-jump state),
    until `t1` is reached (no more events) or `max_events` is exceeded (a
    warning is raised and the partial trajectory is returned).

    `cond_fn` is root-found, so it should be a genuine zero-crossing (e.g.
    `threshold - v`, negative below threshold, crossing zero as `v` reaches
    it) rather than a boolean.
    """
    solver = solver or dfx.Tsit5()
    root_finder = root_finder or optx.Newton(rtol=1e-8, atol=1e-8)
    diffeqsolve_kwargs = dict(diffeqsolve_kwargs or {})

    flow = _as_flow(system)
    term = dfx.ODETerm(flow)
    event = dfx.Event(cond_fn, root_finder)
    seg_saveat = saveat or dfx.SaveAt(steps=True, t1=True)

    ts_chunks: List[jnp.ndarray] = [jnp.asarray([t0])]
    ys_chunks: List[jnp.ndarray] = [jnp.asarray(y0)[None, ...]]
    event_times: List[float] = []

    t, y = t0, y0
    n_events = 0
    while t < t1:
        sol = dfx.diffeqsolve(
            term, solver, t0=t, t1=t1, dt0=dt0, y0=y,
            args=args, event=event, saveat=seg_saveat,
            **diffeqsolve_kwargs,
        )
        seg_ts = jnp.asarray(sol.ts)
        seg_ys = jnp.asarray(sol.ys)
        finite = jnp.isfinite(seg_ts)          # diffrax pads unused save slots with NaN/inf
        seg_ts, seg_ys = seg_ts[finite], seg_ys[finite]

        ts_chunks.append(seg_ts[1:])          # drop the duplicated segment-start point
        ys_chunks.append(seg_ys[1:])

        t_stop = float(seg_ts[-1])
        y_stop = seg_ys[-1]

        # Did this segment end because of the event, or because it simply
        # reached t1 first? `event_mask` is the documented diffrax signal;
        # `t_stop < t1` is a version-agnostic fallback if that attribute
        # isn't present/named differently on your diffrax version.
        event_mask = getattr(sol, "event_mask", None)
        event_fired = (event_mask is not None and bool(jnp.any(jnp.asarray(event_mask)))) \
            or (t_stop < t1 - 1e-9)

        if not event_fired:
            t, y = t_stop, y_stop
            break

        n_events += 1
        if n_events > max_events:
            warnings.warn(
                f"integrate_hybrid: max_events ({max_events}) exceeded before t1={t1}; "
                f"returning the partial trajectory up to t={t_stop}."
            )
            t, y = t_stop, y_stop
            break

        y_jumped = jnp.asarray(jump_fn(t_stop, jnp.asarray(y_stop), args))
        event_times.append(t_stop)
        # Record the post-jump state at the same instant too, so the jump
        # shows up as an explicit discontinuity in the stitched trajectory.
        ts_chunks.append(jnp.asarray([t_stop]))
        ys_chunks.append(jnp.asarray(y_jumped)[None, ...])

        t, y = t_stop, y_jumped

    return HybridSolution(
        ts=jnp.concatenate(ts_chunks),
        ys=jnp.concatenate(ys_chunks, axis=0),
        event_times=jnp.asarray(event_times),
    )


# --------------------------------------------------------------------------
# Common recipe: spiking-neuron voltage threshold/reset + discrete synapses
# --------------------------------------------------------------------------

def make_threshold_condition(
    voltage_indices: Sequence[int],
    threshold: Union[float, Sequence[float]],
) -> CondFn:
    """
    Generic "first spike" event condition over one or more neurons: negative
    while every named voltage is below its own threshold, and crosses zero
    at the first instant ANY one of them reaches it (min-reduction -- valid
    as a root-finding target because generically only one component crosses
    zero at a time; near-simultaneous spikes are still each caught, one per
    event, since `spiking_integrator`'s loop re-solves after every jump).
    """
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
    """
    The common spiking-neuron jump: reset whichever voltage(s) actually
    reached their threshold back to `v_reset` (checked per-neuron at the
    event, so near-simultaneous spikes are each handled correctly), THEN --
    in the SAME jump -- apply `synapse_step_fn` (typically a composite's own
    `.step`, carrying whatever discrete synapse-weight update rule you've
    wired in, e.g. STDP driven by the exact spike time) to the result.
    Pass `synapse_step_fn=None` to reset voltages only, with no synapse update.
    """
    voltage_indices_arr = jnp.asarray(voltage_indices)
    threshold_arr = jnp.broadcast_to(jnp.asarray(threshold, dtype=jnp.float64), voltage_indices_arr.shape)
    v_reset_arr = jnp.broadcast_to(jnp.asarray(v_reset, dtype=jnp.float64), voltage_indices_arr.shape)

    def jump_fn(t, y, args):
        v = y[voltage_indices_arr]
        spiked = v >= threshold_arr
        v_new = jnp.where(spiked, v_reset_arr, v)
        y = y.at[voltage_indices_arr].set(v_new)
        if synapse_step_fn is not None:
            y = synapse_step_fn(t, y, args)
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


# TODO: Diffrax lacks symplectic integrators, implement one
# TODO: same for unitary (split-op routines, ...)

# TODO: some tests and implementations for the method of lines with diffrax
