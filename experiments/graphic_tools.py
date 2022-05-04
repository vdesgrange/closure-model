# Import packages
import numpy as np
import pylab as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import coo_matrix

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
