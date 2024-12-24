#%%
from jax import vmap
from jax import random, lax, numpy as jnp
from jaxtyping import Array
from flax.nnx import Rngs
from jaxtyping import Array


def random_slice(
    x: Array, 
    slice_length: int, 
    slice_count: int,
    rng: Rngs,
) -> tuple[Array, Array]:
    """
    Extracts a random slice of a given length from a given array.

    Args:
        x: Array from which to extract the slice.
        slice_length: Length of the slice to extract.
        rng: Random number generator.
    
    Returns:
        tuple[Array, Array]: A tuple containing the extracted slice and the starting index.
    """
    length = x.shape[0]

    assert length >= slice_length, f"Slice length {slice_length} exceeds array length {length}."

    i = random.randint(rng(), (slice_count, 1), minval=0, maxval=length - slice_length)

    slice = vmap(lax.dynamic_slice, [None, 0, None])(x, i, [slice_length])

    return slice, i


if __name__ == '__main__':
    x = jnp.arange(20)
    rng = Rngs(0)
    window, i = random_slice(x, 3, 8, rng)
    print(window, i)