import numpy as np

def analytical_burgers_1d(t, x, nu):
    """
    An analytical solution to 1D burgers equation.
    Return solution for a single tuple (t, x)
    @param t : time value
    @param x : space value
    @param nu : viscosity constant
    @return u : solution
    """
    r1 = np.exp(-(x - 0.5) / (20 * nu) - (99 * t) / (400 * nu))
    r2 = np.exp(-(x - 0.5) / (4 * nu) - (3 * t) / (16 * nu))
    r3 = np.exp(-(x - 0.375) / (2 * nu))
    return (r1 / 10 + r2 / 2 + r3) / (r1 + r2 + r3)


def get_burgers(t_max, t_min, x_max, x_min, t_n, x_n, nu):
    """
    Compute a Burgers equation solution values for a set of (t, x) tuples.
    """
    t_axis = np.linspace(t_min, t_max, t_n)
    x_axis = np.linspace(x_min, x_max, x_n)
    return analytical_burgers_1d(t_axis[:, None], x_axis[None, :], nu)


def get_burgers_fd(t_max, t_min, x_max, x_min, t_n, x_n, nu, func):
    """
    Compute value of 1D viscous burgers equation for each time step.
    :param u: 2D array with initial values set
    :param S: time discretization size
    :param nu: viscosity parameter
    :param dx: spatial step
    :param dt: time step
    :param func: initialisation function
    """
    dt = (t_max - t_min) / t_n
    dx = (x_max - x_min) / x_n

    u = func(t_max, t_min, x_max, x_min, t_n, x_n, nu)

    for i in range(1, t_n):
        a = nu * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2]) / (dx**2)
        b = 0.25 * u[i-1, 1:-1] * ((u[i-1, 2:] - u[i-1, 0:-2]) / dx)
        u[i, 1:-1] = u[i-1, 1:-1] + dt * (a - b)
    return u


def set_initial_condition(t_max, t_min, x_max, x_min, t_n, x_n, nu):
    """
    Set initial conditions used to compute burgers equation
    :param N: spatial discretization size
    :param L: spatial domain length
    :param S: time discretization size
    :param func: initial function of x at t=0
    """
    dt = (t_max - t_min) / t_n
    dx = (x_max - x_min) / x_n
    u = np.zeros((t_n, x_n))

    u_true = get_burgers(t_max, t_min, x_max, x_min, t_n, x_n - 2, nu)
    u[0, 1:-1] = u_true[0, :]
    u[:, 0] =  u_true[:, 0]
    u[:, -1] = u_true[:, -1]

    return u
