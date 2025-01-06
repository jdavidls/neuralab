#%%

from jax import tree, numpy as jnp  

def check(x, asserts=True):
    valid = tree.map(lambda v: jnp.all(jnp.isfinite(v)) , x)
    valid = tree.reduce(lambda a, b: a & b, valid)
    assert valid, "Dataset contains NaN or infinite values"
    return valid