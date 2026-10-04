from typing import Callable, Tuple, Iterable
import jax.numpy as jnp


# A map should have as inputs the starting coordinates (in some space) and parameters
# As output the coordinates of one iteration of the map

def iterate_map(map_func: Callable[[jnp.ndarray], jnp.ndarray], x0: jnp.ndarray, N: int) -> jnp.ndarray:
    """Iterate a discrete map x_{n+1} = F(x_n) N times.

    Returns array of shape (N+1, dim) including x0.
    """
    x0 = jnp.asarray(x0)
    traj = jnp.zeros((N + 1, x0.size))
    traj[0] = x0
    x = x0.copy()
    for i in range(1, N + 1):
        x = jnp.asarray(map_func(x))
        traj[i] = x
    return traj

# =============================================
# Famous maps
# =============================================

def standard_map(x: jnp.ndarray, k = 0.971635) -> jnp.ndarray:
    # x = [theta, p]
    theta, p = x
    p_new = (p + k * jnp.sin(theta)) % (2 * jnp.pi)
    theta_new = (theta + p_new) % (2 * jnp.pi)
    return jnp.array([theta_new, p_new])

# TODO: Henon
