# Getting Started with Binary Connect

This is a tutorial for *Binary Connect*, a method for training binary neural networks.

## Installation

If you haven't already done so, setup the virtual environment with:

```poetry install```

Activate the virtual environment:

```poetry shell```

Test the example binary connect script with

```python -m bnn.binary_connect_fc_example```

This should train a binary neural network to 92% accuracy on MNIST. It takes about ~20 seconds on my laptop's CPU.

## Introduction to Binary Connect

The key idea of binary connect is that we use real-valued latent weights which describe the probability that the actual weight will be ±1. We use these real-valued latent weights to accumulate gradients, but we sample binary weights from these latent weights at runtime as

$$w_b = Bernoulli(\sigma(w))$$

where we choose to use a "hard sigmoid" function,

$$
\sigma(x) = 
\begin{cases} 
0 & \text{if } x < -1.0, \\
\frac{x + 1}{2} & \text{if } -1.0 \leq x \leq 1.0, \\
1 & \text{if } x > 1.0.
\end{cases}
$$

We can't technically backpropagate through $w_b$ since it is a random variable, so
instead we use take the derivative against the expected value of $w_b$:

$$\frac{\partial E[w_b]}{\partial w} = 1$$

The upshot is that we update $w$ just as if it were an ordinary weight, but we use binary weights during the forward and backward propagation and at test time.

To "freeze" the network for distribution, we can just take any random sample of binary weights.

## Next Steps
- I have made a minimal working example of Binary connect. Please read my code.
  - `binary_connect_utils.py` -- hard sigmoid, sampling, and backprop thru the random weights. (8 LOC)
  - `binary_connect_fc.py` -- creating a binary connect linear layer (17 LOC)
  - `binary_connect_fc_example.py` -- simple example fully connected network using BinaryConnect linear layers.

- Read the original BinaryConnect [paper](https://arxiv.org/pdf/1511.00363.pdf)! It is only 7 pages and it is full of insight. Here are some of my notes on it:
  - Stochastic gradient descent is noisy. It works by averaging out the noise over many steps. Therefore we need sufficient precision to accumulate and average these gradients. So any design will need real-valued latent weights during training.
  - We discretize our weights to {-1, +1}. The discretization is stochastic and can be thought of as adding noise to the true weights. This can actually be beneficial because it works as a kind of regularization. (You can interpret dropout as adding noise to the weights, or rather, multiplying bernoulli noise.)
  - Going from FP32 to Binary is a 32x memory savings. This matters because the bottleneck is often memory: we can't push data through VRAM fast enough. (See [Memory bandwidth wall]()). So memory savings means faster networks.
  - Because we aren't using binary activation functions (I use ReLU in the example), the inputs will be real-valued. Using ±1 for our weights still speeds things up considerably. It turns the usual dot-product multiply-accumulate operation into addition and subtraction. We never have to multiply anything in either forward or backward propagation.
  - Even though we use real-valued latent weights, training is still much faster because we use binary weights for the forward and backward propagation.
  - They discuss backpropagating through the expected value. They show that Dropout and especially DropoutConnect can be interpreted as doing something very similar to BinaryConnect.
  - They give detailed specifications of the fully connected and convolution nets they trained for MNIST and CIFAR.
