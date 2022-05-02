import numpy as np
import torch
from burgers import analytical_burgers_1d
from heat import analytical_heat_1d, get_heat_fd

def gaussian_init(t, x):
    u = np.zeros((t.shape[0], x.shape[0] + 2))
    u[0, 1:-1] = np.exp(-(x - 1)**2)
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def burgers_analytical_init(t, x, nu):
    u = np.zeros((t.shape[0], x.shape[0] + 2))

    u_true = analytical_burgers_1d(t[:, None], x[None, :], nu)
    u[0, 1:-1] = u_true[0, :]
    u[:, 0] =  u_true[:, 0]
    u[:, -1] = u_true[:, -1]

    return u

def random_init(t, x):
    """
    Random initial conditions
    Statistical analysis and simulation of random shocks in stochastic Burgers equation
    Forcing term f = 0
    """
    u = np.zeros((t.shape[0], x.shape[0] + 2))
    nu = np.random.normal(0, 0.25, x.shape[0])
    u[0, 1:-1] = np.sin(x) + nu
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def high_dim_random_init(t, x):
    """
    High-dimensional random initial conditions
    Statistical analysis and simulation of random shocks in stochastic Burgers equation
    """
    m = 48
    nu = np.random.normal(0, 1, 2 * m)
    u = np.zeros((t.shape[0], x.shape[0] + 2))

    s = [(nu[2 * k - 1] * np.sin(k * x)) + (nu[2 * k - 2] * np.cos(k * x)) for k in range(1, m+1)]

    u[0, 1:-1] = (1 / np.sqrt(m)) * np.sum(s, axis=0)
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def heat_analytical_init(t, x, rand=False):
    u0 = np.zeros((t.shape[0], x.shape[0] + 2))
    u, _ = analytical_heat_1d(t[:, None], x[None, :], 50, rand)
    u0[0, 1:-1] = u[0, :]
    u0[:, 0] = 0
    u0[:, -1] = 0

    return u0

def heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, rand=-1, typ=-1):
    if (rand != -1):
        np.random.seed(rand)

    t, dt = np.linspace(t_min, t_max, t_n, retstep=True)
    x, dx = np.linspace(x_min, x_max, x_n, retstep=True)
    
    rand_init = np.random.randint(2)
    if typ > -1:
        rand_init = typ

    init = {
        0: random_init(t, x),
        1: high_dim_random_init(t, x)
    }
    
    u0 = init[rand_init]
    u_df, _ = get_heat_fd(dt, dx, t_n, x_n, u0)
    
    if np.isfinite(u_df).sum() != (u_df.shape[0] * u_df.shape[1]):
        print("not finite.")
        u0 = heat_analytical_init(t, x, False)
        u_df, _ = get_heat_fd(dt, dx, t_n, x_n, u0)
    
    return u_df
