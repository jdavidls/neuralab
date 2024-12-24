from jax import numpy as jnp



def cosine_distance(v1, v2):
    """Calcula la distancia del coseno entre dos vectores."""
    norm1 = jnp.sqrt(jnp.sum(v1 * v1))
    norm2 = jnp.sqrt(jnp.sum(v2 * v2))
    dot_product = jnp.sum(v1 * v2)
    return 1.0 - dot_product / (norm1 * norm2)