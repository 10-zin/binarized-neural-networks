# Experimental training of a bnn using the hard step activation function and modelling activities as Gaussians.

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnn.binary_connect_utils import hard_sigmoid


std_normal = torch.distributions.Normal(0, 1)
def standard_erf(x):
    """
    Standard error function.
    """
    return std_normal.cdf(x)

class GCLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        # initialize the weights to zero
        # because the actual binary weights are randomly sampled, the outputs won't be zero
        self.weight.data = torch.zeros_like(self.weight.data)
        # hook called after each update to clamp the weights to the range [-1, 1]
        self.weight.register_hook(self.clamp_weights)
        # if the inputs and weights are Bernoulli(0.5), then threshold should be input_size/4
        # however in earlier layers the inputs are not Bernoulli(0.5), so we use a lower threshold
        # threshold should be a parameter that is learned
        # TODO this will push all the samples to zero

    def clamp_weights(self, grad):
        """
        clamp the weights to [-1, 1]
        """
        with torch.no_grad():
            self.weight.clamp_(-1, 1)

    def forward(self, input_p, samples):
        """
        input_p is the probability of each input neuron being active.
        samples is a list of samples of input_p. It lets us capture covar.
        """
        # mu is the expected value of the output
        mu = F.linear(input_p, self.weight)  # self.weight is the expected value of the binary weights
        # analytical variance of the output, based on inpendent inputs
        ps = torch.einsum('i,oi->oi', input_p, hard_sigmoid(self.weight))
        vars = 4 * ps * (1 - ps)
        var_indept = torch.einsum('oi->o', vars) + 1e-6
        # empirical variance of the output, based on samples
        with torch.no_grad():
            S, I = samples.shape
            weight_probs = hard_sigmoid(self.weight.repeat((S, 1, 1)))
            binary_weights = 2*torch.bernoulli(weight_probs) - 1
            out_samples = torch.einsum('si,soi->so', samples, binary_weights)
            var_empirical = torch.var(out_samples, dim=0, unbiased=False) + 1e-6
            var_scale = var_empirical / var_indept
            # print('var_scale', var_scale)
        sigma2 = var_scale * var_indept
        return mu, sigma2, out_samples

def gc_hardstep(mu, sigma2, out_samples):

    # apply hardstep
    with torch.no_grad():
        out_samples = torch.where(out_samples > 0, 1., 0.)
    p_out = standard_erf(mu / torch.sqrt(sigma2))
    return p_out, out_samples

# using ±1 instead of 0/1 makes it easier to threshold
# no matter the input distribution, if the weights are Bernoulli(0.5)
# then the output will have mean 0.

# The variance is E[X^2] - E[X]^2 = 1 - (2f - 1)^2 = 4f(1-f)

# We make w_b in {-1, +1}
# We make p in {0, 1}

# ps is the prob that x_i*w_ji = 1
# the var(x_i*w_ji) = 4*ps*(1-ps)