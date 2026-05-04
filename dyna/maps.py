from typing import Callable, Tuple, Iterable
import numpy as np


# A map should have as inputs the starting coordinates (in some space) and parameters
# As output the coordinates of one iteration of the map

def iterate_map(map_func: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, N: int) -> np.ndarray:
    """Iterate a discrete map x_{n+1} = F(x_n) N times.

    Returns array of shape (N+1, dim) including x0.
    """
    x0 = np.asarray(x0)
    traj = np.zeros((N + 1, x0.size))
    traj[0] = x0
    x = x0.copy()
    for i in range(1, N + 1):
        x = np.asarray(map_func(x))
        traj[i] = x
    return traj

def standard_map(x: np.ndarray, k = 0.971635) -> np.ndarray:
    # x = [theta, p]
    theta, p = x
    p_new = (p + k * np.sin(theta)) % (2 * np.pi)
    theta_new = (theta + p_new) % (2 * np.pi)
    return np.array([theta_new, p_new])

# TODO:
# Henon
