import torch.nn as nn

def mse_fn(pred_x, x):
    """
    Mean square error (L2-norm)
    """
    return nn.MSELoss(reduction='mean')(pred_x, x)

def ae_fn(pred_x, x):
    """
    L2-norm summation
    """
    return nn.MSELoss(reduction='sum')(pred_x, x)
