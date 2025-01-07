#%%

from jax import lax, jit
from jax import numpy as jnp, random
from flax import nnx
from typing import Callable, Any, Optional
from jaxtyping import Array, Float

@jit
def ema_fn(
    x: Float[Array, "l ..."],
    decay: Float[Array, "n"],
    state: Optional[Float[Array, "... n"]] = None,
) -> Float[Array, "l ..."]:

    """Efficiently computes Exponential Moving Averages (EMAs) using `lax.scan`.

    This function calculates EMAs for a given ijnput sequence `x` with multiple
    decay rates specified in `decay`. It leverages `lax.scan` for optimized
    recursive computation, enabling efficient processing and automatic
    differentiation.

    Args:
        x: Ijnput array of shape (l, ...), where 'l' is the sequence length
           and '...' represents additional dimensions.
        decay: Array of shape (n,) containing the smoothing factors (decay
               rates) for each EMA. 'n' denotes the number of EMAs to
               compute.
        state: Optional initial state array of shape (... n). If not provided,
               the initial EMA for each decay rate is set to the first element
               of the ijnput sequence.

    Returns:
        A tuple containing:
            - ema_x_z: The final EMA state array of shape (... emas).
            - ema_x: Array of EMAs of shape (l, ... emas).

    Mathematical Formulation:

    The EMA update rule for each decay rate αₖ (where k ∈ {0, ..., emas-1}) is:

    ```latex
    EMA_{i,k} = \alpha_k \times x_i + (1 - \alpha_k) \times EMA_{i-1,k}
    ```

    where:
        - EMA_{i,k} is the EMA at time step i for decay rate αₖ.
        - xᵢ is the ijnput value at time step i.
        - EMA₀ₖ is the initial EMA for decay rate αₖ.
    """
    # Pre-compute initial state
    if state is None:
        state = jnp.broadcast_to(x[0][..., None], x[0].shape + decay.shape)
    
    # Pre-compute coefficients
    alpha = 2.0 / (decay + 1.0)
    _1_alpha = 1.0 - alpha
    
    # Prepare ijnput sequence
    x = x[..., None]  # Add dimension for multiple decay rates
    alpha_x = alpha * x
    
    # Define scan function
    def ema_step(carry, alpha_x_t):
        ema_t = _1_alpha * carry + alpha_x_t
        return ema_t, ema_t
    
    # Run optimized scan
    _, result = lax.scan(ema_step, state, alpha_x)
    
    return result

@jit
def ema_associative_scan(data, alpha):
    n = len(data)
    data += 1e4
    # Calcular las ponderaciones en el espacio logarítmico para estabilidad
    log_weights = jnp.log(1 - alpha) * jnp.arange(n)[::-1]
    
    # Evitar problemas de precisión con valores pequeños (log-sum-exp)
    log_weighted_data = jnp.log(data) + log_weights  # Asegurar log(x_i)
    
    # Escaneo acumulativo logarítmico
    log_num_scan = jnp.cumsum(jnp.exp(log_weighted_data))
    log_den_scan = jnp.cumsum(jnp.exp(log_weights))
    
    # Calcular el EMA dividiendo acumulaciones
    ema = log_num_scan / log_den_scan
    return ema - 1e4

@jit
def ema_with_negatives(data, alpha):
    n = len(data)
    weights = jnp.power(1 - alpha, jnp.arange(n)[::-1])  # Ponderaciones decrecientes
    weighted_data = data * weights  # Datos ponderados
    num_scan = jnp.cumsum(weighted_data)  # Numerador acumulativo
    den_scan = jnp.cumsum(weights)  # Denominador acumulativo
    ema = num_scan / den_scan  # EMA final
    return ema

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    rngs = nnx.Rngs(0)
    # Generate synthetic data
    x = jnp.linspace(0, 10, 100)
    y = jnp.sin(x + 0.1 * random.normal(rngs(), (len(x), )))
    
    # Apply EMA with different alphas
    decays = jnp.array([0.1])
   
    ema = ema_fn(y, decays) # 1.34 ms ± 285 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    assoc_ema_1 = ema_with_negatives(y, .1) # 152 μs ± 74.7 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each) 
    assoc_ema_2 = ema_associative_scan(y, .1) # 84.9 μs ± 44.5 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)

    plt.plot(x, y)
    plt.plot(x, ema)
    plt.plot(x, assoc_ema_1, label="with negs")
    plt.plot(x, assoc_ema_2, label="assoc")
    plt.legend()
    
    

# %%
