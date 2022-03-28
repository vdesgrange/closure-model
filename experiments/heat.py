import numpy as np


def analytical_heat_1d(t, x, n_max: int=1, rand=False):
    """
    Analytical solution to 1D heat equation.
    Return solution for a single tuple (t, x)
    @param t : time value
    @param x : space value
    @param n_max : constant used for computation of heat solution
    @param rand : If true generates, a random vector of constants c to compute heat solution. Else, set all c values at 1.
    @return u : solution
    @return cn : vector of constant c used for computation.
    """
    L = 1.

    cn = np.ones(n_max)
    if rand:
        cn = np.random.rand(n_max)

    u = np.sum([cn[n] * np.exp((-np.pi**2 * n**2 * t) / L) * np.sin((n * np.pi * x) / L) for n in range(n_max)], axis=0)
    return u, cn


def get_heat(t_max, t_min, x_max, x_min, t_n, x_n, rand=False):
    """
    Compute heat equation solution values for a set of (t, x) tuples.
    """
    t_axis = np.linspace(t_min, t_max, t_n)
    x_axis = np.linspace(x_min, x_max, x_n)

    return analytical_heat_1d(t_axis[:, None], x_axis[None, :], 50, rand)


def analytical_grad_t_heat_1d_t(t, x, cn=None, n_max=1):
    """
    Analytical gradient by t of the solution to 1D heat equation
    @param 
    @return : gradient value for a tuple (t, x)
    """

    if cn is None:
        cn = np.ones(n_max)

    return np.sum([-np.pi**2 * n**2 * np.exp(-np.pi**2 * n**2 * t) * np.sin(n * np.pi * x) for n in range(n_max)], axis=0)


def get_heat_grad_t(t_max, t_min, x_max, x_min, t_n, x_n, cn=None):
    """
    Compute gradient by t to heat equation solution values for a set of (t, x) tuples.
    """
    t_axis = np.linspace(t_min, t_max, t_n)
    x_axis = np.linspace(x_min, x_max, x_n)

    return analytical_grad_t_heat_1d_t(t_axis[:, None], x_axis[None, :], cn, 10)


def get_B(X):
    """
    Mass matrix (Gram). Basis matrix. Positive definite, inversible.
    """
    b = np.zeros((X, X))
    d = 1 / (X - 1)

    for i in range(X):
        b[i][i] = 4

    for i in range(X-1):
        b[i][i+1] = 1
        b[i+1][i] = 1

    B = b * d / 6

    return B


def get_A(X):
    """
    Stiffness matrix (Gram).
    """
    a = np.zeros((X, X))
    d = 1 / (X - 1)
    for i in range(X):
        a[i][i] = 2

    for i in range(X-1):
        a[i][i+1] = -1
        a[i+1][i] = -1

    A = a * (1. / d)

    return A


def get_heat_fe(t_max, t_min, x_max, x_min, t_n, x_n):
    """
    Heat solution with finite element method
    @param t_max : t range max value
    @param t_min : t range min value
    @param x_max : x range max value
    @param x_min : x range min value
    @param t_n : number of discrete value on t-axis
    @param x_n : number of discrete value on x-axis
    """
    dt = (t_max - t_min) / t_n
    B = get_B(x_n)  # mass
    A = get_A(x_n)  # stiffness

    c = np.zeros((x_n , t_n))
    u0, _ = get_heat(t_max, t_min, x_max, x_min, t_n, x_n)
    c0 = u0.T
    c[:, 0] = c0[:, 0]

    for t in range(0, t_n - 1):
        tmp = - dt * np.matmul(A, c[:, t])
        c[:, t+1] = np.linalg.solve(B, tmp) + c[:,t]

    return c.T, None


def get_heat_fd(t_max, t_min, x_max, x_min, t_n, x_n):
    """
    Heat solution with finite difference method
    @param t_max : t range max value
    @param t_min : t range min value
    @param x_max : x range max value
    @param x_min : x range min value
    @param t_n : number of discrete value on t-axis
    @param x_n : number of discrete value on x-axis
    """
    dt = (t_max - t_min) / t_n
    dx = (x_max - x_min) / x_n

    ua, _ = get_heat(t_max, t_min, x_max, x_min, t_n, x_n)
    u = np.zeros((t_n, x_n + 1))
    u[0, 1:] = ua[0,:]

    for i in range(1, t_n):
        for k in range(1, x_n):
            u[i][k] = (u[i-1][k-1] - 2 * u[i-1][k] + u[i-1][k+1]) * dt / (dx**2) + u[i-1][k]

    return u, None


if __name__ == '__main__':
    u_true, cn = get_heat(0.2, 0., 1., 0., 20, 50, True)
    g_u_true = get_heat_grad_t(0.2, 0., 1., 0., 20, 50, cn)
    u_fe, _ = get_heat_fe(0.0001, 0.0, 0.5, 0.0, 20, 50)
    u_fd, _ = get_heat_fd(0.0001, 0.0, 0.5, 0.0, 20, 50)
