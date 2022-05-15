# Import packages
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import coo_matrix
from torchdiffeq import odeint_adjoint as odeint

def simple_plotter(ks, title='Simple plot', L=1.0):
    """
    Graphic tool to plot simple values
    """
    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.title(title)

    for k in ks:
        x = np.linspace(0, L, len(k))
        ax.plot(x, k)

    plt.show()


def double_plotter(u1, u2, x_max, x_min, x_n, t_idx=[0, -1], label_1='Expected', label_2='Real', title='Double plotter'):
    fig, ax = plt.subplots(dpi=200)
    fig.tight_layout()
    plt.title(title)

    for t in t_idx:
        x = np.linspace(x_min, x_max, x_n)
        ax.plot(x, u1[t, :], label='{} t={}'.format(label_1, t))
        ax.plot(x, u2[t, :], label='{} t={}'.format(label_2, t))

    plt.legend()
    plt.show()


def animate_plot(u, x):
    fig, ax = plt.subplots(dpi=100)
    xdata, ydata = x, u[0,:]
    ln, = ax.plot(xdata, ydata)

    def init():
        u_max, u_min = np.max(u), np.min(u)
        x_min, x_max = x[0], x[-1]
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(u_min, u_max)
        return ln,

    def update(n):
        xdata = x
        ydata = u[n, :]
        ln.set_data(xdata, ydata)
        return ln,

    frames = np.arange(0, u.shape[0], 1)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=50, blit=True, repeat=True)
    return ani


def show_state(a, title, x='x', y='t', lim=None, aspect=1):
    """
    Graphic tool to show an image.
    This function was implemented to visualize unordered scatter data
    as this function isn't optimized in matplotlib.
    """
    _, axes = plt.subplots(1, 1, figsize=(16, 5), dpi=200)
    im = axes.imshow(a, origin='upper', cmap='inferno', extent=lim, aspect=aspect)
    plt.colorbar(im);
    plt.xlabel(x);
    plt.ylabel(y);
    plt.title(title)
    plt.show()

def show_err(a, title, x='x', y='t', lim=None, aspect=1):
    """
    Graphic tool to show an image.
    This function was implemented to visualize unordered scatter data
    as this function isn't optimized in matplotlib.
    """
    _, axes = plt.subplots(1, 1, figsize=(16, 5), dpi=200)
    im = axes.imshow(a, origin='upper', cmap='YlGnBu', extent=lim, vmin=0., vmax=1., aspect=aspect)
    plt.colorbar(im);
    plt.xlabel(x);
    plt.ylabel(y);
    plt.title(title)
    plt.show()


def matrix_plotter(M):
    """
    Graphic tool to plot the matrix M
    """
    fig, ax = plt.subplots(dpi=150)
    ax.spy(M)

    if (type(M) is not coo_matrix):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if (M[i][j]): 
                    color = 'w' if M[i][j] != -2 else 'k'
                    ax.text(i, j, str(int(M[i][j])), va='center', ha='center', color=color)

    plt.show()


def domain_curve(k, v, L):
    """
    Graphic tool to plot eigenvector v in 1-dimension domain.
    :param k: eigenvalue associated to the eigenvector
    :param v: eigenvector
    :param L: number of row
    :param title: plot title
    """
    fig, ax = plt.subplots()
    fig.tight_layout()
    frequency = np.sqrt(abs(np.real(k)))
    plt.title(r'Eigenmode for $\lambda$={:2f}'.format(frequency))
    ax.plot(v, 'bo')
    plt.show()

def visualize_u_from_F(F, t, u, u0):
    pred_u = odeint(F, u0, t).detach().numpy() # time axis is inversed compare to u_true
    show_state(u.T, 'Real', 't', 'x', None)
    show_state(pred_u.T, 'Determined', 't', 'x', None)

def visualize_F_with_u(F, t_n=20, x_n=100):
    u, x = np.mgrid[-1.:1.:10j, 0.:1.:10j]
    grid_xu = torch.from_numpy(np.stack([x, u], -1).reshape(x_n * x_n, 2)).float()
    t = torch.Tensor(np.linspace(0., 1.0, t_n))
    u_m = np.zeros((t_n, x_n * x_n, 2))

    for i in range(1, t_n):
        tmp = F(0, grid_xu).detach().numpy()
        u_m[i] = tmp
    
    color = ['b','r','g','y']
    width=200
    height=150
    
    for i in [5]: # range(1, t_n, 5):
        xlims = (u_m[i, :, 0].min(), u_m[i, :, 0].max())
        ylims = (u_m[i, :, 1].min(), u_m[i, :, 1].max())
        dx = xlims[1] - xlims[0]
        dy = ylims[1] - ylims[0]
        
        buffer = np.zeros((height+1, width+1))
        
        for j, p in enumerate(u_m[i, :, :]):
            x0 = int(round(((p[0] - xlims[0]) / dx) * width))
            y0 = int(round((1 - (p[1] - ylims[0]) / dy) * height))
            buffer[y0, x0] += 0.3
            if buffer[y0, x0] > 1.0: buffer[y0, x0] = 1.0
        
        ax_extent = list(xlims)+list(ylims)
        plt.figure(dpi=150)
        plt.imshow(
            buffer,
            vmin=0,
            vmax=1, 
            cmap=plt.get_cmap('hot'),
            interpolation='lanczos',
            aspect='auto',
            extent=ax_extent)
            
    return u_m
