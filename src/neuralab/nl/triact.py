#%%
from typing import Optional
from jax import numpy as np
from jax import nn
from jax import jit
from numpy import stack

@jit
def triact(x, *, square_root=False, stack_axis: Optional[int] = -1):
    n = nn.relu(-np.tanh(x))
    z = 1 / np.cosh(x)
    p = nn.relu(np.tanh(x))

    if square_root:
        result = n, z, p
    else:
        result = np.square(n), np.square(z), np.square(p)

    if stack_axis is not None:
        result = np.stack(result, axis=stack_axis)

    return result


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    x = np.linspace(-6, 6, 101)

    n, z, p = triact(x, stack_axis=None)

    plt.plot(x, n, label="$n = relu(\\tanh{x})^2$")
    plt.plot(x, z, label="$z = sech({x})^2$")
    plt.plot(x, p, label="$p = relu(-\\tanh {x})^2$")
    plt.plot(x, n + z + p, label="n + z + p")
    plt.legend()
