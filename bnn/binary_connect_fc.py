import torch
import torch.nn as nn 
import torch.nn.functional as F

from bnn.binary_connect_utils import BCSampler

class BCLinear(nn.Linear):
    """
    Linear layer with binary weights and activations implementing BinaryConnect.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        # initialize the weights to zero
        # because the actual binary weights are randomly sampled, the outputs won't be zero
        self.weight.data = torch.zeros_like(self.weight.data)
        # hook called after each update to clamp the weights to the range [-1, 1]
        self.weight.register_hook(self.clamp_weights) 

    def clamp_weights(self, grad):
        """
        Since the gradient is zero outside of the range [-1, 1],
        clamp the weights to this range after each update.
        """
        with torch.no_grad():
            self.weight.clamp_(-1, 1)

    def forward(self, input):
        """
        Randomly sample {-1, +1} using the hard sigmoid function to get bit probabilities from the weights.
        We expect x*w to be normally distributed N(0, σ²=in_features).
        The paper says to use Batch Normalization, but I just divide by σ=sqrt(in_features) for simplicity.
        """
        binary_weights = BCSampler.apply(self.weight)  # sample {-1, +1}
        out = F.linear(input, binary_weights, self.bias)  # x*w + b
        out /= self.in_features**0.5  # divide by σ=sqrt(in_features)
        return out