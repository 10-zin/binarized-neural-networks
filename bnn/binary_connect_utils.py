import torch

def hard_sigmoid(x):
    """
    Piecewise linear approximation of sigmoid function.
    This is used to map a weight between -1 and 1 to a probability that it is equal to +1,
    such that the expected value E[w_b] = w.
    0          if x < -1.0
    (x+1)/2    if -1.0 <= x <= 1.0
    1          if x > 1.0
    """
    return torch.clamp((x+1.)/2., 0, 1)

class BCSampler(torch.autograd.Function):
    """
    Samples from a Bernoulli distribution using the hard sigmoid function.
    We can't backprop through a random sampling operation,
    so we backpropagate against the expected value of our sample.
    We constructed the hard_sigmoid function such that the expected value of the sample is equal to the input.
    Therefore we just pass along the gradients.
    """
    def forward(ctx, input):
        # inputs (in [-1,1]) represent the pre-activated bit probabilities
        # ctx is a context object that can be used to stash information, but we don't need it
        # return -1 or +1, sampling from a Bernoulli distribution
        # with p = hard_sigmoid(input)
        # return torch.where(input > 0, 1., -1.)  # doesn't speed things up
        return torch.where(torch.bernoulli(hard_sigmoid(input)) == 1, 1., -1.)

    def backward(ctx, grad_output):
        return grad_output
