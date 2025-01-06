from typing import Optional
from jax import numpy as jnp, lax
from jaxtyping import Array, Float

def ema(
    x: Float[Array, "L ..."],
    decay: Float[Array, "()"],
    init: Optional[Float[Array, "..."]] = None,
) -> Float[Array, "L ..."]:

    """Efficiently computes Exponential Moving Averages (EMAs) using `lax.scan`.

    This function calculates EMAs for a given input sequence `x` with multiple
    decay rates specified in `decay`. It leverages `lax.scan` for optimized
    recursive computation, enabling efficient processing and automatic
    differentiation.

    Args:
        x: Input array of shape (l, ...), where 'l' is the sequence length
           and '...' represents additional dimensions.
        decay: Array of shape (emas,) containing the smoothing factors (decay
               rates) for each EMA. 'emas' denotes the number of EMAs to
               compute.
        state: Optional initial state array of shape (... emas). If not provided,
               the initial EMA for each decay rate is set to the first element
               of the input sequence.

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
        - xᵢ is the input value at time step i.
        - EMA₀ₖ is the initial EMA for decay rate αₖ.
    """
    if init is None:
        init = x[0]
        init = jnp.repeat(init, decay.shape[0], axis=-1)

    alpha = 2 / (decay + 1)

    def ema_step(ema_0, x_1):
        ema_1 = alpha * x_1 + (1 - alpha) * ema_0
        return ema_1, ema_1

    _, result = lax.scan(ema_step, init, x[..., None])

    return result
