import numpy as np
from burgers import analytical_burgers_1d

def gaussian_init(t, x):
    u = np.zeros((t.shape[0], x.shape[0] + 2))
    u[0, 1:-1] = np.exp(-(x - 1)**2)
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def analytical_init(t, x, nu):
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