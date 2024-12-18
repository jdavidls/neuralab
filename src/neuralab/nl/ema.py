# %%
from typing import Callable, Optional
from jax import numpy as np
from jax import nn, random
from jax import jit
from jax import lax
from numpy import stack
from flax import nnx

from jaxtyping import Float, Array

EPS = 1e-5
def ema_scan(input: Float[Array, "length"], decay: Float, *, state: Float = None):

    def ema_step(x, value):
        value = decay * value + (1 - decay) * x
        return value, value

    if state is None:
        state = input[0]

    return lax.scan(ema_step, state, input)


def uniform_decay(seed: nnx.Rngs):
    def init():
        return random.uniform(seed.params(), minval=EPS, maxval=1-EPS)
    return init

class EMA(nnx.Module):
    def __init__(self, init_decay: Float):
        self.decay = nnx.Param(init_decay)
        self.state = nnx.Variable(None)

    def __call__(self, x):
        decay = self.decay.value
        state = self.state.value

        state, result = ema_scan(x, decay, state=state)
        self.state.value = state
        return result
    
    @staticmethod
    def stack(count: int):
        @nnx.vmap
        def stack_emas(decay):
            return EMA(decay)
        return stack_emas

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    rngs = nnx.Rngs(3)
    ema = EMA(0.5)
    u = np.cumsum(random.normal(random.PRNGKey(3), (100,)))
    y = ema(u)

    plt.plot(u, label="u")
    plt.plot(y, label="y")
#%%
nnx.vmap(EMA, in_axes=0, out_axes=0)([{
    "decay": np.linspace(0, 1, 10),
    "rngs": nnx.Rngs(3),
}])
# %%
