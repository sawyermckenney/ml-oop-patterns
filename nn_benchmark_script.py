#You should run this script on a system with a GPU or a TPU
#This was tested on Colab with TPUs
import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx

import torch
import torch.nn as nn

class JAXMLP(eqx.Module):
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

class TorchMLP(nn.Module):
    def __init__(self, in_dim, width, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def benchmark_jax_train(model, x, y):
    #calculate the gradient using JAX
    @jax.jit
    def step(model, x, y):
        def loss_fn(model, x, y):
            pred = jax.vmap(model)(x)
            return jnp.mean((pred - y) ** 2)
        return jax.grad(loss_fn)(model, x, y)

    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # initialize and account for the JIT set up time 
    # This allows for JAX to trace and make the XLA Machine code
    grads = step(model, x, y)
    jax.block_until_ready(grads)
    
    # Run the model one more time to see how fast the NN is without compilation time
    start = time.time()
    grads = step(model, x, y)
    jax.block_until_ready(grads)
    return time.time() - start

def benchmark_torch_train(model, x, y):
    # account for initilization time to make the comparison fair
    # Pytorch has low init time but this ensure we remove all compilation time from the final time
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()
    model.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    # Run the model one more time to see how fast the NN is without compilation time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start = time.time()
    model.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return time.time() - start

def run_scaling_benchmark():
    key = jax.random.PRNGKey(0)
    in_dim = 512
    width = 1024
    out_dim = 10
    
    batch_sizes = [2048, 4096, 8192, 16384, 32768, 65536]

    print(f"{'Batch Size'} | {'JAX (s)'} | {'PyTorch (s)'}")
    print("-" * 50)

    for size in batch_sizes:
        # Create random data to be passed through the Equinox NN
        k1, k2 = jax.random.split(key)
        x_jax = jax.random.normal(k1, (size, in_dim))
        y_jax = jax.random.normal(k2, (size, out_dim))
        
        # Create random data to be passed through the pyTorch NN
        x_torch = torch.randn(size, in_dim)
        y_torch = torch.randn(size, out_dim)
        
        #initialize the Models
        jax_model = JAXMLP(in_dim, width, out_dim, key)
        torch_model = TorchMLP(in_dim, width, out_dim)

        # Run the models 10 times and display the mean time it took to process the batch
        N = 10
        jax_times = []
        torch_times = []
        for _ in range(N):
            jax_times.append(benchmark_jax_train(jax_model, x_jax, y_jax))
            torch_times.append(benchmark_torch_train(torch_model, x_torch, y_torch))      

        print(f"{size} | {np.mean(jax_times)} | {np.mean(torch_times)}")

run_scaling_benchmark()