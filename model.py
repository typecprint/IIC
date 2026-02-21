import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys


class ModelIIC(nn.Module):
    def __init__(self, num_classes=10):
        super(ModelIIC, self).__init__()
        # Use updated torchvision weights
        self.model_conv = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Freeze all layers
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # Replace classification layer
        num_ftrs = self.model_conv.classifier[1].in_features
        self.model_conv.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, num_classes, bias=True),
        )

    def forward(self, x):
        # Apply softmax across classes
        x = F.softmax(self.model_conv(x), dim=1)
        return x


def compute_joint(phi_x, phi_x_prime):
    """
    Computes the joint probability matrix p_cc.
    """
    p_cc = phi_x.unsqueeze(2) * phi_x_prime.unsqueeze(1)
    p_cc = p_cc.sum(dim=0)
    p_cc = (p_cc + p_cc.t()) / 2.0  # symmetrise
    p_cc = p_cc / p_cc.sum()  # normalize
    return p_cc


def IID_loss(phi_x, phi_x_prime, EPS=sys.float_info.epsilon):
    """
    IIC (Invariant Information Clustering) Loss implementation.
    Maximizes mutual information between predicted cluster assignments of original and transformed images.
    """
    _, k = phi_x.shape
    p_cc = compute_joint(phi_x, phi_x_prime)  # k x k matrix

    p_c = p_cc.sum(dim=1).view(k, 1).expand(k, k)
    p_c_prime = p_cc.sum(dim=0).view(1, k).expand(k, k)

    loss = -p_cc * (torch.log(p_cc) - torch.log(p_c) - torch.log(p_c_prime))

    # Mask near-zero probabilities using epsilon
    p_cc = torch.where(
        p_cc < EPS, torch.tensor(EPS, device=p_cc.device, dtype=p_cc.dtype), p_cc
    )
    p_c = torch.where(
        p_c < EPS, torch.tensor(EPS, device=p_c.device, dtype=p_c.dtype), p_c
    )
    p_c_prime = torch.where(
        p_c_prime < EPS,
        torch.tensor(EPS, device=p_c_prime.device, dtype=p_c_prime.dtype),
        p_c_prime,
    )

    loss = loss.sum()
    return loss
