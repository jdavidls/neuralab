# %%
from typing import Optional, Tuple
from einops import repeat
from jax import numpy as np
from jax import random
from jax import lax
from jax import nn
from flax import nnx
from math import pi as PI
from jaxtyping import Float, Array

def ema_scan(
    x: Float[Array, "l ..."], 
    decay: Float[Array, "emas"],
    state: Optional[Float[Array, "... emas"]] = None,
) -> Tuple[
    Float[Array, "... emas"], 
    Float[Array, "l ... emas"]
]:
    emas = len(decay)

    if state is None:
        state = repeat(x[0], "... -> ... emas", emas=emas) 

    print(state.shape)

    def ema_step(ema_0, x_1):

        ema_1 = decay * x_1 + (1 - decay) * ema_0

        return ema_1, ema_1

    return lax.scan(ema_step, state, x[..., None])


class EMA(nnx.Module):
    def __init__(self, emas: int, rngs: nnx.Rngs):
        self.log_decay: nnx.Param[Float[Array, "emas"]] = nnx.Param(
            np.linspace(-PI, PI, emas)
        )

        self.state: nnx.Variable[Optional[Float[Array, "... emas"]]] = nnx.Variable(
            None
        )

    @property
    def decay(self):
        return nn.sigmoid(self.log_decay.value)

    def reset(self):
        self.state.value = None

    def __call__(self, x):

        ema_x_z, ema_x = ema_scan(
            x, 
            self.decay,
            self.state.value
        )

        self.state.value = ema_x_z

        return ema_x

    def stats(self, x):

        x2 = np.square(x)

        emas = self(np.stack([x, x2], axis=1))

        ema_x, ema_x2 = emas[:, 0], emas[:, 1]

        # exponential standard deviation
        esd = np.sqrt(ema_x2 - np.square(ema_x))

        return ema_x, esd

    def standarize(self, x):
        ema_x, esd = self.stats(x)

        return (x[..., None] - ema_x) / (esd + (1-self.decay))

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    EMAS = 4
    rngs = nnx.Rngs(3)
    ema = EMA(EMAS, rngs=rngs)
    u = np.cumsum(random.normal(random.PRNGKey(3), (250,)))
    ea, esd = ema.stats(u)
    std = ema.standarize(u)

    #plt.plot(u, label="u")
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

#%%
