import numpy as np
import torch
import torch.nn.functional as F


def gan_loss(output, label, noise_std=0.0, flip_threshold=0.0):
    target = torch.full_like(output, label)
    if noise_std != 0.0:
        noise = torch.randn_like(target) * noise_std 
        target += noise

    if flip_threshold != 0.0:
        if np.random.rand() < flip_threshold:
            target = 1 - target
    return F.mse_loss(output, target)
