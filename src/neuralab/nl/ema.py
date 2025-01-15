# %%
from typing import Optional, Tuple

from einops import repeat
from flax import nnx
from jax import lax, nn
from jax import numpy as np
from jax import random
from jaxtyping import Array, Float

from neuralab.nl.common import State
from neuralab import fn

from typing import Optional
from jax import numpy as jnp, lax
from jaxtyping import Array, Float

def ema_fn(
    x: Float[Array, "L ..."],
    decay: Float[Array, "E"],
    ema_init: Optional[Float[Array, "..."]] = None,
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
    # Pre-compute initial state
    if ema_init is None:
        ema_init = jnp.broadcast_to(x[0][..., None], x[0].shape + decay.shape)
    
    # Pre-compute coefficients
    alpha = 2.0 / (decay + 1.0)
    alpha_complement = 1.0 - alpha
    
    # Prepare input sequence
    x_expanded = x[..., None]  # Add dimension for multiple decay rates
    x_scaled = alpha * x_expanded
    
    # Define scan function
    def ema_step(carry, x_t):
        ema_t = alpha_complement * carry + x_t
        return ema_t, ema_t
    
    # Run optimized scan
    _, result = lax.scan(ema_step, ema_init, x_scaled)
    
    return result


def emav_fn(
    x: Float[Array, "L ..."],
    decay: Float[Array, "E"],
    ema_init: Optional[Float[Array, "..."]] = None,
    emv_init: Optional[Float[Array, "..."]] = None,
) -> tuple[Float[Array, "L ..."], Float[Array, "L ..."]]:
    """Efficiently computes Exponential Moving Average (EMA) and Variance (EMV) using `lax.scan`.

    This function calculates both the exponential moving average and its corresponding
    variance for a given input sequence. The implementation uses JAX's `lax.scan` for
    efficient computation.

    Args:
        x: Input array of shape (L, ...) where L is the sequence length
        decay: Decay factor array of shape (). Larger values result in slower decay.
              The actual decay rate alpha is computed as 2/(decay + 1)
        ema_init: Optional initial EMA value. If None, initialized with first value of x
        emv_init: Optional initial EMV value. If None, initialized with zeros

    Returns:
        A tuple containing:
        - ema: Exponential Moving Average array of shape (L, ...)
        - emv: Exponential Moving Variance array of shape (L, ...)

    Implementation details:
        The EMA is computed as: ema_t = (1-α)·ema_{t-1} + α·x_t
        The EMV is computed as: emv_t = (1-α)·emv_{t-1} + α·(x_t - ema_t)²
        where α = 2/(decay + 1)
    """
    if ema_init is None:
        ema_init = x[0]
        ema_init = jnp.repeat(ema_init, decay.shape[0], axis=-1)

    if emv_init is None:
        emv_init = jnp.zeros_like(ema_init)

    alpha = 2 / (decay + 1)
    _1_alpha = 1 - alpha

    def emav_step(carry, x_1):
        ema_0, emv_0 = carry

        ema_1 = _1_alpha * ema_0 + alpha * x_1
        emv_1 = _1_alpha * emv_0 + alpha * (x_1 - ema_1) ** 2

        return (ema_1, emv_1), (ema_1, emv_1)

    _, (ema, emv) = lax.scan(emav_step, (ema_init, emv_init), x[..., None])

    return ema, emv


class EMStats(nnx.Module):
    """Class for managing and computing Exponential Moving Averages (EMAs).

    This class encapsulates the EMA computation, standard deviation estimation,
    and standardization functionalities. It maintains internal state for efficient
    processing of sequential data.

    Attributes:
        log_decay: Parameter storing the logarithm of decay rates. Transformed
                   using sigmoid to obtain actual decay values between 0 and 1.
        state: Variable holding the current EMA state. Initialized to None and
               updated during computation.
    """

    def __init__(self, num_layers: int, min_t=2, max_t=512, learn=True):
        self.num_layers = num_layers
        self.training = False
        self.min_t = min_t
        self.max_t = max_t
        
        self.scale = max_t - min_t

        log_decay_init = np.linspace(-np.pi, np.pi, num_layers)

        self.log_decay = (
            nnx.Param(log_decay_init) if learn else nnx.Cache(log_decay_init)
        )

        self.state: State[Optional[Float[Array, "... num_layers"]]] = State(None)

    @property
    def decay(self):
        """Returns the sigmoid-transformed decay values."""
        return nn.sigmoid(self.log_decay.value) * self.scale + self.min_t

    def reset(self):
        """Resets the internal EMA state."""
        self.state.value = None

    def ema(self, x):
        """Computes EMAs for the given input.

        Args:
            x: Input array.

        Returns:
            Array of computed EMAs.
        """

        ema_init = self.state.value
        if ema_init is None:
            ema_init = np.mean(x[: self.max_t // 2], axis=0)
            ema_init = repeat(ema_init, "... -> ... l", l=self.num_layers)

        ema_x = ema_fn(x, self.decay, ema_init=ema_init)
        # if self.training is False:
        # self.state.value = ema_x[-1]
        return ema_x

    def emav(self, x):
        """Computes the exponential moving average and variance of the input.

        Args:
            x: Input array.

        Returns:
            Tuple containing the EMA and EMV.
        """
        # x2 = np.square(x)
        # emas = self(np.stack([x, x2], axis=1))
        # ema, ema_x2 = emas[:, 0], emas[:, 1]
        
        ema_init = np.mean(x[: self.max_t // 2], axis=0)
        ema_init = repeat(ema_init, "... -> ... l", l=self.num_layers)

        emv_init = np.var(x[: self.max_t // 2], axis=0)
        emv_init = repeat(emv_init, "... -> ... l", l=self.num_layers)

        ema, emv = emav_fn(x, self.decay, ema_init=ema_init, emv_init=emv_init)
        
        # x2 = np.square(x)
        
        # emas = self(np.stack([x, x2], axis=1))
        # ema, ema_x2 = emas[:, 0], emas[:, 1]


        # Exponential moving variance
        #emv = (ema2 - ema**2) #*  2 / (decay + 1)

        return ema, emv

    def norm(self, x, eps=1e-4):
        """Standardizes the input using EMA and ESD.

        Args:
            x: Input array.

        Returns:
            Standardized input array.


        Mathematical Formulation:

        ```latex
        Standardized_x = \frac{x - EMA(x)}{ESD + (1 - \alpha)}
        ```

        The `(1 - α)` term in the denominator provides numerical stability,
        preventing division by zero when ESD is close to zero.
        """
        ema, emv = self.emav(x)
        nrm = (x[..., None] - ema) * lax.rsqrt(emv + eps)
        return ema, emv, nrm


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from neuralab.trading.dataset import Dataset

    ds = Dataset.load()

    L = 500
    p = ds.log_price[:L, 0,0]
    u = ds.log_volume[:L,0,0]
    v = ds.log_imbalance[:L,0,0]

    num_layers = 4
    emstats = EMStats(num_layers)
    # u = np.cumsum(random.normal(random.PRNGKey(3), (L,)))
    u_ema = emstats.ema(u)
    _, u_emv = emstats.emav(u)
    _, _, u_nrm = emstats.norm(u)

    v_ema = emstats.ema(v)
    _, v_emv = emstats.emav(v)
    _, _, v_nrm = emstats.norm(v)

    # plt.plot(u, label="u")
    plt.title("EM Average")
    # plt.plot(u[:L], label=f"true")
    for n in range(num_layers):
        plt.plot(u_ema[:L, n], label=f"avg(u) {emstats.decay[n]:.2f}")
        plt.plot(v_ema[:L, n], label=f"avg(v) {emstats.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Variance")
    for n in range(num_layers):
        plt.plot(u_emv[:L, n], label=f"var(u) {emstats.decay[n]:.2f}")
        plt.plot(v_emv[:L, n], label=f"var(v) {emstats.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Standarization")
    for n in range(num_layers):
        plt.plot(u_nrm[:L, n], label=f"norm(u) {emstats.decay[n]:.2f}", alpha=0.5)
        plt.plot(v_nrm[:L, n], label=f"norm(v) {emstats.decay[n]:.2f}", alpha=0.5)
    plt.legend()
    plt.show()

    # %%
    p_avg, p_var = emstats.emav(p)
    p_std = emstats.norm(p)
    # plt.plot(u, label="u")
    plt.title("EM Average")
    # plt.plot(u[:L], label=f"true")
    for n in range(num_layers):
        plt.plot(p_avg[:L, n], label=f"avg(u) {emstats.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Variance")
    for n in range(num_layers):
        plt.plot(p_var[:L, n], label=f"var(u) {emstats.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Standarization")
    for n in range(num_layers):
        plt.plot(p_std[:L, n], label=f"norm(u) {emstats.decay[n]:.2f}", alpha=0.5)

    plt.legend()
    plt.show()

# %%
