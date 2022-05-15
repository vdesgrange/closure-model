import numpy as np
import torch
from equations.burgers import analytical_burgers_1d 

def gaussian_init(t, x):
    u = np.zeros((t.shape[0], x.shape[0]))
    u[0, :] = np.exp(-(x - 1)**2)
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def burgers_analytical_init(t, x, nu):
    u = np.zeros((t.shape[0], x.shape[0]))

    u_true = analytical_burgers_1d(t[:, None], x[None, :], nu)
    u[0, :] = np.copy(u_true[0, :])
    u[:, 0] =  0 # u_true[:, 0]
    u[:, -1] = 0 # u_true[:, -1]

    return u

def random_init(t, x):
    """
    Random initial conditions
    Statistical analysis and simulation of random shocks in stochastic Burgers equation
    Forcing term f = 0
    """
    #u = np.zeros((t.shape[0], x.shape[0] + 2))
    u = np.zeros((t.shape[0], x.shape[0]))
    nu = np.random.normal(0, 0.25, x.shape[0])
    u[0, :] = np.sin(x) + nu
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
    #u = np.zeros((t.shape[0], x.shape[0] + 2))
    u = np.zeros((t.shape[0], x.shape[0]))

    s = [(nu[2 * k - 1] * np.sin(k * x)) + (nu[2 * k - 2] * np.cos(k * x)) for k in range(1, m+1)]

    u[0, :] = (1 / np.sqrt(m)) * np.sum(s, axis=0)
    u[:, 0] =  0
    u[:, -1] = 0

    return u

def analytical_heat_1d(t, x, n=[], c=[], k=1.):
    """
    Analytical solution to 1D heat equation.
    Return solution for a single tuple (t, x)
    @param t : time value
    @param x : space value
    @param c :
    @param n :
    @return u : solution
    @return cn : vector of constant c used for computation.
    """
    L = 1.
    if len(c) == 0:
        c = np.divide(np.random.normal(0., 1., len(n)), n) # n = range(1, n_max), 1 to avoid dividing by 0

    u = np.sum([c[i] * np.exp(- k * (np.pi * n[i] / L)**2 * t) * np.sqrt(2 / L) * np.sin(n[i] * np.pi * x / L) for i in range(len(n))], axis=0)
    return u, c

def heat_analytical_init(t, x, n=[], c=[], k=1.):
    u0 = np.zeros((t.shape[0], x.shape[0]))
    u, _ = analytical_heat_1d(t[:, None], x[None, :], n, c, k)
    u0[0, :] = np.copy(u[0, :])
    u0[:, 0] = 0
    u0[:, -1] = 0

    return u0
