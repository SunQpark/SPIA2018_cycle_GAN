import torch
import torch.nn.functional as F


def gan_loss(output, label):
    target = torch.full_like(output, label)
    return F.mse_loss(output, target)

# def recon_l1_loss(output, target)