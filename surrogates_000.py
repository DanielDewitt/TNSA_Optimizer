import torch
from torch import nn
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt



class SolenoidSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = 1000
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class TSolenoidSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = 100
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, self.nodes),
            nn.PReLU(),
            nn.Linear(self.nodes, 4),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

