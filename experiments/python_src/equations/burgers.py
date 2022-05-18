import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate

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


def get_burgers_fd(dt, dx, t_n, nu, u0):
    """
    Compute value of 1D non-conservative viscous burgers equation for each time step.
    :param nu: viscosity parameter
    """
    u = np.copy(u0)

    for i in range(1, t_n):
        a = nu * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2]) / (dx**2)
        b = 0.25 * u[i-1, 1:-1] * ((u[i-1, 2:] - u[i-1, 0:-2]) / dx)
        u[i, 1:-1] = u[i-1, 1:-1] + dt * (a - b)
    return u


def get_burgers_cons_fd(dt, dx, t_n, nu, u0):
    """
    Compute value of 1D conservative viscous burgers equation for each time step.
    :param nu: viscosity parameter
    """
    u = np.copy(u0)
    
    f = lambda u : np.power(u, 2) / 2
    
    for i in range(1, t_n):
        a  = (nu / dx**2) * (u[i-1, 2:] - 2 * u[i-1, 1:-1] + u[i-1, 0:-2])
        b = ( 1 / (2 * dx)) * (f(u[i-1, 2:]) - f(u[i-1, 0:-2]))
        u[i, 1:-1] = u[i-1, 1:-1] + dt * (a - b)
    
    return u


def get_D(X, s):
    d = np.zeros((X, X))
    for i in range(X):
        d[i][i] = 1 - s
        
    for i in range(X-1):
        d[i][i+1] = s / 2
        d[i+1][i] = s / 2
    
    return d


def get_M(X, u, dt, dx, nu):
    M = np.zeros((X, X))
    s = nu * dt / dx**2
    b = 1 + s
    
    for i in range(X):
        M[i][i] = b
    
    for i in range(X-1):
        M[i+1][i] = -dt / (4 * dx) * u[i-1] - s / 2
        M[i][i+1] = dt / (4 * dx) * u[i+1] - s / 2
    
    return M


def get_burgers_nicolson(dt, dx, t_n, x_n, nu, u0):
    """
    Crank-Nicolson algorithm for 1D non-conservative viscous burgers equation.
    """
    s = nu * dt / dx**2
    d = get_D(x_n, s)

    u = np.zeros((t_n, x_n))
    u[0, :] = np.copy(u0[0, 1:-1])

    for i in range(1, t_n):
        b = d @ u[i-1, :]
        m = get_M(x_n, u[i-1, :], dx, dt, nu)
        u[i, :] = la.solve(m, b)

    return u

def get_burgers_fft(t, dx, x_n, nu, u0, method="LSODA"):
    """
    @param t: t evaluation step
    @param dx: spacial step
    @param x_n: number spatial discretization steps
    @param nu: viscosity
    @param u0: initial conditions (t_n by x_n matrix)
    @param method: method to use. Default LSODA (odeint default). BDF recommanded.
    """
    k = 2 * np.pi * np.fft.fftfreq(x_n, d=dx)

    def burgers_pde(t, u, k, nu): # t, u for odeint, u,t for solve_ivp
        # FFT - Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * k * u_hat # First differential
        u_hat_xx = -k**2 * u_hat # Second differential

        # IFFT - Spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)

        u_t = -u * u_x + nu * u_xx
        return u_t.real

    u_rad =  integrate.solve_ivp(burgers_pde, t_span=(t[0], t[-1]), y0=u0[0, :], t_eval=t, method=method, args=(k, nu))
    # u_rad = integrate.odeint(burgers_pde, u0[0, 1:-1], t, args=(k, nu,)) # LSODA
    return u_rad.y.T # Just u_rad for odeint

def set_initial_condition(t_max, t_min, x_max, x_min, t_n, x_n, nu):
    """
    Set initial conditions used to compute burgers equation
    :param N: spatial discretization size
    :param L: spatial domain length
    :param S: time discretization size
    :param func: initial function of x at t=0
    """
    u = np.zeros((t_n, x_n))

    u_true = get_burgers(t_max, t_min, x_max, x_min, t_n, x_n - 2, nu)
    u[0, 1:-1] = u_true[0, :]
    u[:, 0] =  u_true[:, 0]
    u[:, -1] = u_true[:, -1]

    return u