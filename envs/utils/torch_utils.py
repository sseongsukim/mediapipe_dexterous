import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, dict):
        x = {k: x[k].cpu().detach().numpy() for k in x.keys()}

    return x.cpu().detach().numpy()


def to_torch(x, device=None):
    if device is not None:
        return torch.from_numpy(x).to(device)
    else:
        return torch.from_numpy(x)
