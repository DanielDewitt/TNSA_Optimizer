import torch
from torch import nn


class WMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float64, requires_grad=True)
    def forward(self,input,target):
        return torch.mean(((input - target)**2 ) * self.weight)