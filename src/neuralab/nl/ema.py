# %%
from typing import Optional, Tuple

from einops import repeat
from flax import nnx
from jax import lax, nn
from jax import numpy as np
from jax import random
from jaxtyping import Array, Float

from neuralab.nl.common import State


def ema_fn(
    x: Float[Array, "l ..."],
    decay: Float[Array, "emas"],
    init: Optional[Float[Array, "... emas"]] = None,
    *,
    is_stationary=False,
) -> Tuple[
    Float[Array, "... emas"],
    Float[Array, "l ... emas"],
]:
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
    emas = len(decay)

    if init is None:  # TODO: si el canal es estacionario el estado inicial debe ser 0
        init = x[0]
        if is_stationary:
            init = np.zeros_like(init)
        init = repeat(init, "... -> ... emas", emas=emas)

    def ema_step(ema_0, x_1):
        ema_1 = decay * x_1 + (1 - decay) * ema_0
        return ema_1, ema_1

    ema_z, ema = lax.scan(ema_step, init, x[..., None])

    return ema_z, ema


class EMA(nnx.Module):
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

    def __init__(self, emas: int, rngs: nnx.Rngs):
        self.log_decay: nnx.Param[Float[Array, "emas"]] = nnx.Param(
            np.linspace(-np.pi, np.pi, emas)
        )
        self.state: State[Optional[Float[Array, "... emas"]]] = State(None)

    @property
    def decay(self):
        """Returns the sigmoid-transformed decay values."""
        return nn.sigmoid(self.log_decay.value)

    def reset(self):
        """Resets the internal EMA state."""
        self.state.value = None

    def __call__(self, x, is_stationary=False):
        """Computes EMAs for the given input.

        Args:
            x: Input array.

        Returns:
            Array of computed EMAs.
        """
        ema_x_z, ema_x = ema_fn(
            x, self.decay, self.state.value, is_stationary=is_stationary
        )
        # self.state.value = ema_x_z
        return ema_x

    def emasd(self, x, eps=1e-6, is_stationary=False):
        """Computes EMA and Exponential Standard Deviation (ESD).

        Args:
            x: Input array.

        Returns:
            Tuple containing the EMA and ESD.
        """
        x2 = np.square(x)
        emas = self(np.stack([x, x2], axis=1), is_stationary=is_stationary)
        ema, ema_x2 = emas[:, 0], emas[:, 1]
        

        # Exponential standard deviation
        esd = np.sqrt(eps + ema_x2 - np.square(ema))

        return ema, esd
    
    def emav(self, x, eps=1e-6, is_stationary=False):
        """Computes the exponential moving average and variance of the input.

        Args:
            x: Input array.

        Returns:
            Tuple containing the EMA and EMV.
        """
        x2 = np.square(x)
        emas = self(np.stack([x, x2], axis=1), is_stationary=is_stationary)
        ema, ema_x2 = emas[:, 0], emas[:, 1]

        # Exponential moving variance
        emv = eps + ema_x2 - np.square(ema)

        return ema, emv

    def standarize(self, x, eps=1e-6, is_stationary=False):
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
        ema, esd = self.emasd(x, eps=eps, is_stationary=is_stationary)

        # Standarize the input
        # ema_normalized = (x[..., None] - ema + eps) / (eps + esd * (1+self.decay))
        x = x[..., None]
        ema_normalized = (x - ema) / (esd + eps + (1 - self.decay))

        return ema_normalized


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from dataproxy.dataset import DATASETS

    L = 100
    p = np.log(DATASETS[0]["vwap"].to_numpy())
    u = np.log(DATASETS[0]["vol"].to_numpy())
    v = (
        np.log(DATASETS[0]["bid_vol"].to_numpy())
        - np.log(DATASETS[0]["ask_vol"].to_numpy())
    )

    EMAS = 3
    rngs = nnx.Rngs(3)
    ema = EMA(EMAS, rngs=rngs)
    # u = np.cumsum(random.normal(random.PRNGKey(3), (L,)))
    u_ema, u_esd = ema.emasd(u)
    u_ems = ema.standarize(u)

    v_ema, v_esd = ema.emasd(v, is_stationary=True)
    v_ems = ema.standarize(v, is_stationary=True)

    # plt.plot(u, label="u")
    plt.title("Exponential Moving Average")
    #plt.plot(u[:L], label=f"true")
    for n in range(EMAS):
        plt.plot(u_ema[:L, n], label=f"ema(u) {ema.decay[n]:.2f}")
        plt.plot(v_ema[:L, n], label=f"ema(v) {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("Exponential Standard Deviation")
    for n in range(EMAS):
        plt.plot(u_esd[:L, n], label=f"esd(u) {ema.decay[n]:.2f}")
        plt.plot(v_esd[:L, n], label=f"esd(v) {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("Exponential Moving Standarization")
    for n in range(EMAS):
        plt.plot(u_ems[:L, n], label=f"ems(u) {ema.decay[n]:.2f}")
        plt.plot(v_ems[:L, n], label=f"ems(v) {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    # %%
    from neuralab.nl import triact

    # plt.plot(u_ems[:, 0])
    plt.pcolormesh((triact(v_ems[:, 0]) * u_ems[:, 0, None]).T)
    # plt.plot(np.log(p), label="u_ems")
    plt.show()

    #%%
    plt.hist(u_ems[:, 1], bins=100)
    plt.show()