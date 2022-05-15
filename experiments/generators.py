import numpy as np
import torch
from equations.initial_functions import random_init, high_dim_random_init, heat_analytical_init, burgers_analytical_init
from utils.analysis_tools import downsampling
from equations.heat import get_heat_fd_impl, get_heat_fft
from equations.burgers import get_burgers_fft
from utils.graphic_tools import show_state

def get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, rand=-1, typ=-1, d=1):
    # Compute a snapshot of the solution u(t, x).
    u_s = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, rand, typ, d)
    u_s = torch.from_numpy(u_s).float()
    t = torch.from_numpy(np.linspace(t_min, t_max, t_n)).float()
    u0 = np.copy(u_s[0, :])
    return t, u0, u_s

def heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, rand=-1, typ=-1, d=1):
    if (rand != -1):
        np.random.seed(rand)

    t, dt = np.linspace(t_min, t_max, t_n, retstep=True)
    x, dx = np.linspace(x_min, x_max, x_n, retstep=True)
    
    rand_init = np.random.randint(2)
    if typ > -1:
        rand_init = typ

    init = {
        0: random_init,
        1: high_dim_random_init,
        2: (lambda a, b: heat_analytical_init(a, b, list(range(1, 51)), [], True))
    }
    
    u0 = np.copy(init[rand_init](t, x))
    # u_df, _ = get_heat_fd_impl(dt, dx, t_n, x_n, u0)
    u_df = get_heat_fft(t, dx, x_n, d, u0)
    
    if np.isfinite(u_df).sum() != (u_df.shape[0] * u_df.shape[1]):
        print("not finite.")
        u0 = np.copy(heat_analytical_init(t, x, False))
        # u_df, _ = get_heat_fd_impl(dt, dx, t_n, x_n, u0)
        u_df = get_heat_fft(t, dx, x_n, d, u0)
    
    return u_df

def get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, rand=-1, downsize=0):
    t_batch_size = t_n

    # Compute a snapshot of the solution u(t, x).
    u_s = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, rand, -1)

    if downsize > 0:
        u_s = downsampling(u_s, downsize)
        t_batch_size = t_n // downsize
    u_s = torch.from_numpy(u_s).float()

    t = np.linspace(t_min, t_max, t_batch_size)
    batch_t = torch.from_numpy(t).float()
    batch_u0 = u_s[0, :]
    #batch_u = torch.stack([u_s[i, :] for i in range(0, t_batch_size)], dim=0)
    batch_u = u_s

    return batch_t, batch_u0, batch_u

def burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, nu, rand=-1, typ=-1):
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
    u_df = get_burgers_fft(t, dx, x_n, nu, u0, method="BDF")

    if np.isfinite(u_df).sum() != (u_df.shape[0] * u_df.shape[1]):
        print("not finite.")
        u0 = burgers_analytical_init(t, x, nu)
        u_df = get_burgers_fft(t, dx, x_n, nu, u0, method="BDF")

    return u_df

if __name__ == "__main__":
    t_max = 0.2
    t_min = 0.01
    x_max = 1.
    x_min = 0.
    t_n = 64
    x_n = 64
    _, _, bu = get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, -1, 0)
    show_state(bu[1:,:], 'bu')
