import torch

def hard_sigmoid(x):
    """
    Piecewise linear approximation of sigmoid function
    0          if x < -1.0
    (x+1)/2    if -1.0 <= x <= 1.0
    1          if x > 1.0
    """
    return torch.clamp((x+1.)/2., 0, 1)

class BCSampler(torch.autograd.Function):
    """
    Samples from a Bernoulli distribution using the hard sigmoid function
    """
    @staticmethod
    def forward(ctx, input):
        # inputs represent the pre-activated bit probabilities
        # ctx is a context object that can be used to stash information, but we don't need it
        # return -1 or +1, sampling from a Bernoulli distribution
        # with p = hard_sigmoid(input)
        return torch.where(torch.bernoulli(hard_sigmoid(input)) == 1, 1., -1.)

    @staticmethod
    def backward(ctx, grad_output):
        # return the gradient of the hard sigmoid function
        return grad_output
