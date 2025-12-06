import jax 
import jax.numpy as jnp
from jax import random, grad, jit


# This is a two layer Neural Network in JAX, what it does is it takes 10 
# input features which are the features of the data (age, gender, etc.)
# and it has 5 hidden nodes which are the hidden nodes of the Neural Network
# and it has 1 output node which is the output of the Neural Network
# the output is a sigmoid function which is a logistic function
# the sigmoid function is a function that takes a real number and squashes it to a value between 0 and 1.

def init_params(key):
    w1_key, w2_key = random.split(key)
    return {
        "w1": random.normal(w1_key, (10, 5)),
        "w2": random.normal(w2_key, (5, 1)),
    }
def forward(params, x):
    x = jnp.dot(x, params["w1"])
    x = jnp.maximum(x, 0) 
    return jax.nn.sigmoid(jnp.dot(x, params["w2"]))
key = random.PRNGKey(0)
params = init_params(key)
output = forward(params, jnp.ones((1, 10)))
print(output.item())