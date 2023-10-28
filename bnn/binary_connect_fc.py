import torch
import torch.nn as nn 
import torch.nn.functional as F

from bnn.binary_connect_utils import BCSampler

class BCLinear(nn.Linear):
    """
    Linear layer with binary weights and activations
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.bc_sampler = BCSampler.apply
        self.weight.data = torch.zeros_like(self.weight.data) #.uniform_(-1, 1)
        self.weight.register_hook(self.clamp_weights)  # hook called after each update to the weights

    def clamp_weights(self, grad):
        with torch.no_grad():
            self.weight.clamp_(-1, 1)

    def forward(self, input):
        return F.linear(input, self.bc_sampler(self.weight) / self.in_features**0.5, self.bias)