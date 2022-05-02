import numpy as np
import torch
from analysis_tools import downsampling
from initial_functions import heat_snapshot_generator


def generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, n=256, typ=1, filename='heat_training_set.pt'):
    train_set = []

    for _ in range(n):
        t, dt = np.linspace(t_min, t_max, t_n * 8, retstep=True)
        high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n * 8, x_n * 8, rand=-1, typ=typ)
        low_dim = downsampling(high_dim, 8)
        low_t = np.array([(i * 3.5) * dt for i in range(t_n)])

        batch_low_t = torch.from_numpy(low_t).float()
        batch_low_dim = torch.from_numpy(low_dim).float()
        batch_high_t = torch.from_numpy(t).float()
        batch_high_dim = torch.from_numpy(high_dim).float()

        item = [batch_low_t, batch_low_dim, batch_high_t, batch_high_dim]
        train_set.append(item)
    torch.save(train_set, filename)

    return train_set

def get_heat_batch(t_max, t_min, x_max, x_min, t_n, x_n, x_batch_size, rand=-1, downsize=0):
    t_batch_size = t_n
        
    # Compute a snapshot of the solution u(t, x).
    u_s = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n, x_n, rand)
    if downsize > 0:
        u_s = downsampling(u_s, downsize)
        t_batch_size = t_n // downsize
    u_s = torch.from_numpy(u_s).float()
    
    t = np.linspace(t_min, t_max, t_batch_size)
    batch_t = torch.from_numpy(t).float()
    batch_u0 = u_s[0, :]
    batch_u = torch.stack([u_s[i, :] for i in range(0, t_batch_size)], dim=0)
    # batch_u = u_s
    
    return batch_t, batch_u0, batch_u

def read_dataset(filepath='heat_training_set.pt'):
    return torch.load(filepath)
