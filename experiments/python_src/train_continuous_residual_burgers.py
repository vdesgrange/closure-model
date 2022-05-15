# Import packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

from graphic_tools import simple_plotter, show_state, visualize_u_from_F, show_err
from burgers import get_burgers, get_burgers_fd, get_burgers_cons_fd, get_burgers_nicolson, get_burgers_fft
from generators import burgers_snapshot_generator, get_burgers_batch
from initial_functions import random_init, high_dim_random_init, burgers_analytical_init
from analysis_tools import relative_err, rmse, check_weights, downsampling
from training_dataset import generate_burgers_training_dataset, read_dataset
from models import BurgersModelA


def training_ode_solver_net(net, epochs, t_n, x_n, dataset, val_epoch=10, rands=[], downsize=0):
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    training_batch_size = len(dataset)
    training_set_idx = np.arange(0, training_batch_size)
    
    tr_min_t = 1
    tr_max_t = int(len(dataset[0][0]) / 5 * 3)
    val_max_t = int(tr_max_t + len(dataset[0][0]) / 5 * 2)
    
    for e in range(1, epochs + 1):
        loss_tot = 0
        val_loss_tot = 0
        np.random.shuffle(training_set_idx)
        
        # === Train ===
        net.train()
        for i in training_set_idx:
            # === Randomness ====
            rand = -1
            if (i < len(rands)):
                rand = rands[i]
            
            optimizer.zero_grad()
            
            t, bu, _, _ = dataset[i]
            tr_t =  t[tr_min_t:tr_max_t]
            tr_b0 = bu[tr_min_t, :]
            tr_bu = bu[tr_min_t:tr_max_t, :]
            
            pred_u = odeint(net, tr_b0, tr_t) # [1:-1]
            loss = loss_fn(pred_u.T, tr_bu.T) # [:, 1:-1]
            loss_tot += loss.item()
            
            loss.backward(retain_graph=False) # retain_graph=True if 2+ losses
            optimizer.step()
        
        print('Epoch %d loss %f'%(e, float(loss_tot / float(training_batch_size))))
        
        # === Evaluate ===
        net.eval()
        if (e > val_epoch):
            for j in training_set_idx:
                t, bu, _, _ = dataset[j]
                val_t = t[tr_max_t:val_max_t]
                val_b0 = bu[tr_max_t, :]
                val_bu = bu[tr_max_t:val_max_t, :]
                
                val_pred_u = odeint(net, val_b0, val_t)
                val_loss = loss_fn(val_pred_u.T, val_bu.T)
                val_loss_tot += val_loss.item()
            print('Epoch %d validation loss %f'%(e, float(val_loss_tot / float(training_batch_size))))
        
        
        if e % 10 == 0:
            sample_t, sample_b0, sample_real = get_burgers_batch(t_max, t_min, x_max, x_min, t_n, x_n, nu, rand, -1)
            sample_b0 = sample_real[1, :]
            sample_pred = odeint(net, sample_b0, sample_t[1:])
            show_state(sample_real[1:].T, 'Real', 't', 'x', None)
            show_state(sample_pred.detach().numpy().T, 'Determined', 't', 'x', None)
    
    return net


def main():
    net = BurgersModelA(x_n)
    F = training_ode_solver_net(net, 50, t_n, x_n, training_set[0:64], 5, [], -1)
    pass


if __name__ == "__main__":
    main()
