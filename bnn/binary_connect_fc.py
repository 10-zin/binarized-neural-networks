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
        self.weight.register_hook(self.clamp_weights)  # hook called after each update to the weights
        # initialize weights to zero
        # I didn't notice much difference between uniform and zero initialization
        self.weight.data = torch.zeros_like(self.weight.data) #.uniform_(-1, 1)
        # alpha is the scaling factor for the weights (same for all weights)
        self.alpha = 1/torch.sqrt(torch.tensor(self.in_features, dtype=torch.float))

    def clamp_weights(self, grad):
        with torch.no_grad():
            self.weight.clamp_(-1, 1)

    def forward(self, input):
        return F.linear(input, self.alpha * self.bc_sampler(self.weight), self.bias)