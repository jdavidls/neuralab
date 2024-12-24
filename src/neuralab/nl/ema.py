# %%
from math import pi as PI
from typing import Optional, Tuple

from einops import repeat
from flax import nnx
from jax import lax, nn
from jax import numpy as np
from jax import random
from jaxtyping import Array, Float

from neuralab.nl.common import State


def ema_scan(
    x: Float[Array, "l ..."],
    decay: Float[Array, "emas"],
    state: Optional[Float[Array, "... emas"]] = None,
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

    if state is None:
        state = repeat(x[0], "... -> ... emas", emas=emas)

    def ema_step(ema_0, x_1):
        ema_1 = decay * x_1 + (1 - decay) * ema_0
        return ema_1, ema_1

    return lax.scan(ema_step, state, x[..., None])


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
            np.linspace(-PI, PI, emas)
        )
        self.state: State[
            Optional[Float[Array, "... emas"]]
        ] = State(None)

    @property
    def decay(self):
        """Returns the sigmoid-transformed decay values."""
        return nn.sigmoid(self.log_decay.value)

    def reset(self):
        """Resets the internal EMA state."""
        self.state.value = None

    def __call__(self, x):
        """Computes EMAs for the given input.

        Args:
            x: Input array.

        Returns:
            Array of computed EMAs.
        """
        ema_x_z, ema_x = ema_scan(x, self.decay, self.state.value)
        # self.state.value = ema_x_z
        return ema_x

    def stats(self, x):
        """Computes EMA and Exponential Standard Deviation (ESD).

        Args:
            x: Input array.

        Returns:
            Tuple containing the EMA and ESD.
        """
        x2 = np.square(x)
        emas = self(np.stack([x, x2], axis=1))
        ema_x, ema_x2 = emas[:, 0], emas[:, 1]

        # Exponential standard deviation
        esd = np.sqrt(1e-4 + ema_x2 - np.square(ema_x))

        return ema_x, esd

    def standarize(self, x):
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
        ema_x, esd = self.stats(x)

        
        # Standarize the input
        ema_normalized = (x[..., None] - ema_x + 1e-4) / (1e-4 + esd + (1 - self.decay))

        return ema_normalized


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    EMAS = 4
    rngs = nnx.Rngs(3)
    ema = EMA(EMAS, rngs=rngs)
    u = np.cumsum(random.normal(random.PRNGKey(3), (250,)))
    ea, esd = ema.stats(u)
    std = ema.standarize(u)

    # plt.plot(u, label="u")
    for n in range(EMAS):
        plt.plot(ea[:, n], label=f"ema {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    for n in range(EMAS):
        plt.plot(esd[:, n], label=f"esd {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    for n in range(EMAS):
        plt.plot(std[:, n], label=f"std {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

