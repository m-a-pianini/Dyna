"""
saltation.py
=============

The saltation matrix: the correct linearization of a state-triggered jump
for propagating TANGENT (variational) vectors across it -- needed whenever
you estimate Lyapunov exponents of a hybrid system (e.g. the spiking-neuron
+ synapse composites this project builds) via the augmented-state /
Benettin-QR approach already used in `benettin.flow_spectrum`.

Why not just use the Jacobian of the jump map?
-------------------------------------------------
Because the event TIME is itself a function of the state: a perturbed
trajectory crosses the switching surface `{cond_fn(x) = 0}` at a slightly
different time than the reference trajectory. `D(jump_fn)` alone only
accounts for "the state changed", not "and so the crossing happened earlier
or later, which itself changes what state actually gets fed into the jump".
The saltation matrix is the correction that captures both effects together;
see the module-level docstring in `dyna/integrators/integrators.py` for the
`cond_fn`/`jump_fn` conventions this builds on.

    S = DR(x-) + (f+ - DR(x-) f-) . grad(g)(x-)^T / (grad(g)(x-) . f-)

    R = jump_fn, g = cond_fn, f- = flow just before the event,
    f+ = flow just after (i.e. at the post-jump state).

Reference: di Bernardo, Budd, Champneys & Kowalczyk, "Piecewise-smooth
Dynamical Systems" (2008); Zillmer, Livi, Politi & Torcini (2006) for the
spiking-neuron-network Lyapunov-spectrum use case this project targets.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = ["saltation_matrix"]

FlowFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]   # (t, y, args) -> dy
CondFn = Callable[..., Any]                               # (t, y, args, **kwargs) -> scalar
JumpFn = Callable[[Any, jnp.ndarray, Any], jnp.ndarray]   # (t, y, args) -> y_new


def saltation_matrix(
    flow: FlowFn,
    jump_fn: JumpFn,
    cond_fn: CondFn,
    t: Any,
    x_minus: jnp.ndarray,
    args: Any,
) -> jnp.ndarray:
    """
    The saltation matrix S at a single event, evaluated at the state `x_minus`
    immediately BEFORE the jump (i.e. exactly where `integrate_hybrid`'s
    `cond_fn` root-finds to zero). Multiply a tangent vector by S -- in place
    of `jax.jacfwd(jump_fn)` -- to correctly carry it across the event.

    `flow`, `jump_fn`, `cond_fn` are exactly the objects you'd already pass
    to `dyna.integrators.integrators.integrate_hybrid` -- this function adds
    no new conventions, so it drops straight into that same setup.

    Returns an (n, n) array, n = x_minus.size.
    """
    x_minus = jnp.asarray(x_minus)

    f_minus = jnp.asarray(flow(t, x_minus, args))
    x_plus = jnp.asarray(jump_fn(t, x_minus, args))
    f_plus = jnp.asarray(flow(t, x_plus, args))

    DR = jax.jacfwd(lambda x: jnp.asarray(jump_fn(t, x, args)))(x_minus)
    dg = jax.grad(lambda x: jnp.asarray(cond_fn(t, x, args)))(x_minus)

    denom = jnp.dot(dg, f_minus)
    correction = jnp.outer(f_plus - DR @ f_minus, dg) / denom

    return DR + correction
