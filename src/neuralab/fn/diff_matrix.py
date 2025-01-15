from jax import numpy as jnp

def diff_matrix(
    x,
):  #
    a = x[..., None, :]
    b = x[..., :, None]
    i, j = jnp.triu_indices(x.shape[-1], 1)
    return (a - b)[..., i, j]

def diff_matrix_num_outputs(num_inputs: int) -> int:
    return (num_inputs * num_inputs - num_inputs) // 2

 