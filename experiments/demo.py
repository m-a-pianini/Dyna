import argparse
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import jax.numpy as jnp
import diffrax as dfx
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)

from dyna.lyapunov.benettin import mLCE_flow, mLCE_map, flow_spectrum
from dyna.maps import iterate_map, standard_map
from time import time

# Parser nonsense
parser = argparse.ArgumentParser(description='Integrator demo: maps and Hamiltonians')
parser.add_argument('--demo', choices=['standard_map', 'pendulum'], default='none')
parser.add_argument('--iters', type=int, default=2000)
parser.add_argument('--k', type=float, default=0.971635)
args = parser.parse_args()

if args.demo == 'standard_map':
    k = args.k*0
    dynamic = lambda x: standard_map(x, k)

    init_theta = np.concat([np.linspace(1.2, 1.7, 10), np.linspace(0, 6, 5)])
    init_p = np.concat([np.linspace(2.5, 3, 10), np.linspace(0, 6, 5)])
    init_values = np.array(list(it.product(init_theta, init_p)))
    #init_values = np.array([np.array([np.pi]*100), np.linspace(0, 2*np.pi, 100)]).transpose()

    lyaps = []
    plt.figure(figsize=(6, 5))

    for i, init in enumerate(init_values):
        #init = np.array([0, init_values[i]])
        traj = iterate_map(dynamic, init, args.iters)
        sim = iterate_map(dynamic, np.array([2*np.pi, 2*np.pi]) - init, args.iters)
        plt.scatter(traj[::1, 0], traj[::1, 1], s=0.5)
        plt.scatter(sim[::1, 0], sim[::1, 1], s=0.5)
        lyaps.append(mLCE_map(dynamic, init, 2000))
        print(f'Estimated mLCE (map) for initial condition {init}: {lyaps[i]:.10f}')

    plt.xlabel('theta')
    plt.ylabel('p')
    plt.title(f'Standard map k={k} ({args.iters} iter)')
    plt.tight_layout()
    plt.show()


elif args.demo == 'pendulum':
    # Simple pendulum with H = p^2/2 - cos(theta)
    def pendulum_flow(t: jnp.ndarray, z: jnp.ndarray, m=1, g=9.81, L=1) -> jnp.ndarray:
        theta, p = z

        return jnp.array([p, -(m*g/L)*jnp.sin(theta)])

    z0 = jnp.array([1.0, 0.0])

    t_bounds = [0, 30]
    delta_t = 1e-4

    solver = dfx.Dopri5()
    term = dfx.ODETerm(lambda t, z, args: pendulum_flow(t, z))

    saveat = dfx.SaveAt(ts=jnp.linspace(t_bounds[0], t_bounds[1], 10000))

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t_bounds[0],
        t1=t_bounds[1],
        dt0=delta_t,
        y0=z0,
        saveat=saveat,
        args=None,
        max_steps=1200000
)

    qs, ps = sol.ys.transpose()
    plt.figure()
    plt.plot(qs[:] % (2 * jnp.pi) - jnp.pi, ps[:], linewidth=0.5)
    plt.xlabel('theta')
    plt.ylabel('p')
    plt.title('Pendulum phase portrait')
    plt.show()

    steps = 0.03
    N_int = 500000
    start = time()
    lyap_f = flow_spectrum(flow=lambda t, z, args: pendulum_flow(t, z), solver=dfx.Dopri5(), z0=z0,
                                dt=delta_t, interval=steps*delta_t, n_intervals=N_int)
    end = time()
    print(f"Elapsed time: {(end - start):.6f}")
    print('Estimated maximal Lyapunov exponent (flow, approx):', lyap_f)
