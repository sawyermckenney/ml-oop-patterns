import jax 
import jax.numpy as jnp
from jax import random, grad, jit

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
print(output)