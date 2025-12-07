# Install equinox using !pip install equinox
import jax
import jax.numpy as jnp
import equinox as eqx

import torch
import torch.nn as nn


class JAXMLP(eqx.Module):
    """
    This is a two layer Neural Network implemented using equinox 
    It implements a relu activation function
    """
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, in_dim, width, out_dim, key):
        k1, k2 = jax.random.split(key, 2)
        object.__setattr__(self, "fc1", eqx.nn.Linear(in_dim, width, key=k1))
        object.__setattr__(self, "fc2", eqx.nn.Linear(width, out_dim, key=k2))

    def __call__(self, x):
        x = jax.nn.relu(self.fc1(x))
        x = self.fc2(x)
        return x