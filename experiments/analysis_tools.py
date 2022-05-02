import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from graphic_tools import show_state


def relative_err(u_a, u_b):
    return np.true_divide(np.abs(u_a - u_b), np.abs(u_a))


def rmse(pred_u, real_u):
    return np.sqrt(np.mean((pred_u - real_u)**2))


def check_efficiency(net, t_max, t_min, x_max, x_min, t_n, x_n, get_batch, n_sim=10, debug=False):
    rmse_tot = 0
    
    for _ in range(n_sim):
        t, b0, bu = get_batch(t_max, t_min, x_max, x_min, t_n, x_n, x_n, False)
        pred_u = odeint(net, b0[1:-1], t).detach().numpy()
        u = bu[:,1:-1].detach().numpy()
        rmse_tot += rmse(pred_u.T, u.T)

        if debug:
            print("RMSE = ", rmse(pred_u.T, u.T))
            show_state(np.abs(pred_u.T - u.T), 'Error NN vs. Real', 't', 'x', None)
            show_state(u.T, 'Real', 't', 'x', None)
            show_state(pred_u.T, 'Expected', 't', 'x', None)
    
    return rmse_tot / n_sim


def check_weights(net):
    for name, param in net.named_parameters():
        if 'weight' in name:
            weights = param.detach().numpy()
            show_state(weights, 'Layer weight')

def downsampling(u, d=64):
    """
    @param u: snapshot
    @param d: interval of downsampling. Default 64
    """
    n, m = u.shape[0] // d, u.shape[1] // d
    d_u = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            d_u[i][j] = np.mean(u[i*d:(i+1)*d, j*d:(j+1)*d])
    
    return d_u
