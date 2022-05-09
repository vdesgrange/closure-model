import numpy as np
import torch
from analysis_tools import downsampling
from generators import heat_snapshot_generator, burgers_snapshot_generator


def generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, n=256, typ=1, filename='heat_training_set.pt'):
    train_set = []

    for i in range(n):
        print('Item ', i)
        t, dt = np.linspace(t_min, t_max, t_n * 4, retstep=True)
        high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n * 4, x_n * 4, rand=-1, typ=typ)
        low_dim = downsampling(high_dim, 4)
        low_t = np.array([(i * 1.5) * dt for i in range(t_n)])

        batch_low_t = torch.from_numpy(low_t).float()
        batch_low_dim = torch.from_numpy(low_dim).float()
        batch_high_t = torch.from_numpy(t).float()
        batch_high_dim = torch.from_numpy(high_dim).float()

        item = [batch_low_t, batch_low_dim, batch_high_t, batch_high_dim]
        train_set.append(item)

    torch.save(train_set, filename)

    return train_set


def generate_burgers_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, nu, n=256, typ=1, filename='burgers_training_set.pt'):
    train_set = []

    for _ in range(n):
        t, dt = np.linspace(t_min, t_max, t_n * 64, retstep=True)
        high_dim = burgers_snapshot_generator(t_max, t_min, x_max, x_min, t_n * 64, x_n * 64, nu, rand=-1, typ=typ)
        low_dim = downsampling(high_dim, 64)
        low_t = np.array([(i * 31.5) * dt for i in range(t_n)])

        batch_low_t = torch.from_numpy(low_t).float()
        batch_low_dim = torch.from_numpy(low_dim).float()
        batch_high_t = torch.from_numpy(t).float()
        batch_high_dim = torch.from_numpy(high_dim).float()

        item = [batch_low_t, batch_low_dim, batch_high_t, batch_high_dim]
        train_set.append(item)
    torch.save(train_set, filename)

    return train_set


def read_dataset(filepath='heat_training_set.pt'):
    return torch.load(filepath)


if __name__ == '__main__':
    t_max = 0.2
    t_min = 0.01
    x_max = 1.
    x_min = 0.
    t_n = 64
    x_n = 64

    training_set = generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, 256, 0, 'random_heat_training_set2.pt')
    # training_set = read_dataet('dataset/random_heat_training_set.pt')
