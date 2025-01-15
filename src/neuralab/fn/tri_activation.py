#%%
from math import pi as PI
from typing import Optional, Tuple

from flax import nnx
from jax import jit, nn
from jax import numpy as np
from jaxtyping import Array, Float

@jit
def triact(
    x: Float[Array, "..."],
    *,
    apply_square: bool = True,
    concat_axis: Optional[int] = -1,
    stack_axis: Optional[int] = -1,
    # scale: Optional[float] = PI,
) -> Float[Array, "... (n z p)"] | Tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    """Computes a three-way activation based on hyperbolic functions.

    This function calculates three components: negative, near-zero, and positive,
    based on transformations of the input `x` using hyperbolic tangent (tanh)
    and hyperbolic secant (sech) functions.

    Args:
        x: Input array.
        apply_square: Whether to square the output components. Defaults to False.
        stack_outputs: Axis to stack outputs (n, z, p). If None, returns a tuple. Defaults to -1.
        input_scale: Scaling factor applied to the input. Defaults to π.

    Returns:
        Stacked array or tuple of arrays (n, z, p), depending on `stack_outputs`.

    Formulas:
        Let `s = input_scale`

        n = relu(-tanh(s * x))          (or n = relu(-tanh(s * x))^2 if apply_square=True)
        z = sech(s * x)                 (or z = sech(s * x)^2 if apply_square=True)
        p = relu(tanh(s * x))           (or p = relu(tanh(s * x))^2 if apply_square=True)

        where:
            relu(x) = max(0, x)
            sech(x) = 1 / cosh(x)
    """

    n = nn.relu(-np.tanh(x))
    z = 1.0 / np.cosh(x)  # sech(x)
    p = nn.relu(np.tanh(x))

    if apply_square:
        n = np.square(n)
        z = np.square(z)
        p = np.square(p)

    if concat_axis is not None:
        return np.concat([n, z, p], axis=concat_axis)
    elif stack_axis is not None:
        return np.stack([n, z, p], axis=stack_axis)
    else:
        return n, z, p


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Import here for better organization

    x = np.linspace(-1, 1, 101)

    n, z, p = triact(x, stack_outputs=None)

    plt.plot(x, n, label="$n = relu(-\\tanh(\\pi x))^2$")
    plt.plot(x, z, label="$z = sech(\\pi x)^2$")
    plt.plot(x, p, label="$p = relu(\\tanh(\\pi x))^2$")
    plt.plot(x, n + z + p, label="n + z + p")

    plt.title("Tri-Activation Function Components")
    plt.xlabel("x")
    plt.ylabel("Activation Value")
    plt.legend()
    plt.grid(True)
    plt.show()
