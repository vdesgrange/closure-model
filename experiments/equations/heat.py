import numpy as np
import scipy.sparse as sp
import scipy.integrate as integrate
from equations.initial_functions import random_init, analytical_heat_1d



def get_heat(t, x, n=[], c=[], k=1.):
    """
    Compute heat equation solution values for a set of (t, x) tuples.
    """

    return analytical_heat_1d(t[:, None], x[None, :], n, c, k)


def analytical_grad_t_heat_1d_t(t, x, cn=None, n_max=1):
    """
    Analytical gradient by t of the solution to 1D heat equation
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


def get_heat_fd(dt, dx, t_n, x_n, u0=None):
    """
    Heat solution with finite difference method
    @param t_max : t range max value
    @param t_min : t range min value
    @param x_max : x range max value
    @param x_min : x range min value
    @param t_n : number of discrete value on t-axis
    @param x_n : number of discrete value on x-axis
    """
    u = np.copy(u0)

    for i in range(1, t_n):
        u[i, 1:-1] = u[i-1, 1:-1] + dt * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1][0:-2]) / (dx**2)

    return u, None


def get_heat_fd_impl(dt, dx, t_n, x_n, u0=None):
    u = np.copy(u0)
    s = dt / dx**2

    d = sp.diags([-s * np.ones(x_n - 1), (1 + 2 * s) * np.ones(x_n), -s * np.ones(x_n - 1)], [-1, 0, 1], format = "csc")
    for i in range(1, t_n):
        u[i, :] = sp.linalg.spsolve(d, u[i-1, :])

    u[:, 0] = 0
    u[:, -1] = 0


    return u, None


def get_heat_fft(t, dx, x_n, d=1, u0=None, method="LSODA"):
    k = 2 * np.pi * np.fft.fftfreq(x_n, d=dx)

    def heat_pde(t, u, k, d): # t, u for odeint, u,t for solve_ivp
        # FFT - Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_xx = -k**2 * u_hat # Second differential

        # IFFT - Spatial domain
        u_xx = np.fft.ifft(u_hat_xx)

        u_t =  d * u_xx
        return u_t.real

    u_rad =  integrate.solve_ivp(heat_pde, t_span=(t[0], t[-1]), y0=u0[0, :], t_eval=t, method=method, args=(k, d))
    return u_rad.y.T


if __name__ == '__main__':
    t_max = 0.2
    t_min = 0.01
    x_max = 1.
    x_min = 0.
    t_n = 64
    x_n = 64

    t, dt = np.linspace(t_min, t_max, t_n, retstep=True)
    x, dx = np.linspace(x_min, x_max, x_n, retstep=True)

    u0 = random_init(t, x)
    u_fd, _ = get_heat_fd(dt, dx, t_n, x_n, u0)
    u_fd_impl, _ = get_heat_fd_impl(dt, dx, t_n, x_n, u0)
