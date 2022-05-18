import numpy as np
import torch
from utils.analysis_tools import downsampling
from generators import heat_snapshot_generator, burgers_snapshot_generator


def generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, n=256, typ=1, d=1, filename='heat_training_set.pt'):
    train_set = []

    for i in range(n):
        print('Item ', i)
        t, _ = np.linspace(t_min, t_max, t_n * 4, retstep=True)
        high_dim = heat_snapshot_generator(t_max, t_min, x_max, x_min, t_n * 4, x_n * 4, rand=-1, typ=typ, d=d)
        low_dim = downsampling(high_dim, 4)
        low_t = np.linspace(t_min, t_max, t_n)

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
        # low_t = np.array([(i * 31.5) * dt for i in range(t_n)])
        low_t = np.linspace(t_min, t_max, t_n * 64)

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


def process_dataset(dataset):
    """
    Dataset processing (method A)
    Split provided dataset into a training and validation set split according to time discretization
    @param dataset: list of tuples [low_dim_t, low_dim_u, high_dim_t, high_dim_u]
    @return tr_dataset, val_dataset: dataset split according to time discretization
    """
    batch_size = len(dataset)
    dataset_idx = np.arange(0, batch_size)
    tr_min_t = 0
    tr_max_t = int(len(dataset[0][0]) / 5 * 4)
    val_max_t = int(tr_max_t + len(dataset[0][0]) / 5 * 1)
    
    h_tr_min_t = 0
    h_tr_max_t = int(len(dataset[0][2]) / 5 * 4)
    h_val_max_t = int(h_tr_max_t + len(dataset[0][2]) / 5 * 1)
    
    training_set = []
    validation_set = []
    
    for i in dataset_idx:
        t, bu, ht, hbu = dataset[i]
        # t = torch.from_numpy(np.linspace(0., 0.5, len(dataset[0][0]))).float()
        
        tr_t =  t[tr_min_t:tr_max_t]
        tr_bu = bu[tr_min_t:tr_max_t, :]
        
        val_t =  t[tr_max_t:val_max_t]
        val_bu = bu[tr_max_t:val_max_t, :]
        
        h_tr_t =  ht[h_tr_min_t:h_tr_max_t]
        h_tr_bu = hbu[h_tr_min_t:h_tr_max_t, :]
        
        h_val_t =  ht[h_tr_max_t:h_val_max_t]
        h_val_bu = hbu[h_tr_max_t:h_val_max_t, :]
        
        training_set.append([tr_t, tr_bu, h_tr_t, h_tr_bu])
        validation_set.append([val_t, val_bu, h_val_t, h_val_bu])
        
    return training_set, validation_set


if __name__ == '__main__':
    t_max = 0.2
    t_min = 0.01
    x_max = 1.
    x_min = 0.
    t_n = 64
    x_n = 64

    training_set = generate_heat_training_dataset(t_max, t_min, x_max, x_min, t_n, x_n, 256, 0, 'random_heat_training_set2.pt')
    # training_set = read_dataset('dataset/random_heat_training_set.pt')
