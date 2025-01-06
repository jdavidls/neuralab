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

    def __init__(self, num_layers: int, min_t=2, max_t=512, learn=False):
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
        return nn.sigmoid(self.log_decay.value)**2 * self.scale + self.min_t

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

        init = self.state.value
        if init is None:
            init = np.mean(x[: self.max_t // 2], axis=0)
            init = repeat(init, "... -> ... l", l=self.num_layers)

        ema_x = fn.ema(x, self.decay, init)
        # if self.training is False:
        # self.state.value = ema_x[-1]
        return ema_x

    def avgvar(self, x, is_stationary=False):
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
        emv = ema_x2 - np.square(ema)

        return ema, emv

    def std(self, x, eps=1e-4, is_stationary=False):
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
        ema, emv = self.avgvar(x, is_stationary=is_stationary)
        return (x[..., None] - ema) * lax.rsqrt(emv)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from neuralab.trading.dataset import Dataset

    ds = Dataset.load()

    L = 500
    p = ds.log_price[:L, 0,0]
    u = ds.log_volume[:L,0,0]
    v = ds.log_imbalance[:L,0,0]

    num_layers = 4
    ema = EMA(num_layers)
    # u = np.cumsum(random.normal(random.PRNGKey(3), (L,)))
    u_avg, u_var = ema.avgvar(u)
    u_std = ema.std(u)

    v_avg, v_var = ema.avgvar(v, is_stationary=True)
    v_std = ema.std(v, is_stationary=True)

    # plt.plot(u, label="u")
    plt.title("EM Average")
    # plt.plot(u[:L], label=f"true")
    for n in range(num_layers):
        plt.plot(u_avg[:L, n], label=f"avg(u) {ema.decay[n]:.2f}")
        plt.plot(v_avg[:L, n], label=f"avg(v) {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Variance")
    for n in range(num_layers):
        plt.plot(u_var[:L, n], label=f"var(u) {ema.decay[n]:.2f}")
        plt.plot(v_var[:L, n], label=f"var(v) {ema.decay[n]:.2f}")
    plt.legend()
    plt.show()

    plt.title("EM Standarization")
    for n in range(num_layers):
        plt.plot(u_std[:L, n], label=f"std(u) {ema.decay[n]:.2f}", alpha=0.5)
        plt.plot(v_std[:L, n], label=f"std(v) {ema.decay[n]:.2f}", alpha=0.5)
    plt.legend()
    plt.show()

    # %%
    from neuralab.nl import triact

    # plt.plot(u_ems[:, 0])
    plt.pcolormesh((triact(v_std[:, 0]) * u_std[:, 0, None]).T)
    # plt.plot(np.log(p), label="u_ems")
    plt.show()

    # %%
    plt.hist(v_std[:, 2], bins=100)
    plt.show()

    # %%
    x = np.linspace(1, 10, 100)
    decay = lambda x: 2 / (x + 1)

    # for n in range(1,5):
    plt.plot(x, decay(x), label="decay(x)")
    plt.legend()
    plt.show()


# %%
