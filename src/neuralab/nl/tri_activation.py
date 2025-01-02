#%%
from math import pi as PI
from typing import Optional, Tuple

from flax import nnx
from jax import jit, nn
from jax import numpy as np
from jaxtyping import Array, Float

from neuralab.utils import optional


@jit
def triact(
    x: Float[Array, "..."],
    *,
    apply_square: bool = True,
    stack_outputs: Optional[int] = -1,
    input_scale: Optional[float] = PI,
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

    s = input_scale if input_scale is not None else 1.0  # Default scale = 1.0

    scaled_x = x * s

    n = nn.relu(-np.tanh(scaled_x))
    z = 1.0 / np.cosh(scaled_x)  # sech(x)
    p = nn.relu(np.tanh(scaled_x))


    if apply_square:
        n = np.square(n)
        z = np.square(z)
        p = np.square(p)

    if stack_outputs is not None:
        result = np.stack([n, z, p], axis=stack_outputs)  # List comprehension is slightly cleaner
    else:
        result = (n, z, p)

    return result

class TriactFeed(nnx.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None, *, rngs: nnx.Rngs):

        out_features = optional(out_features, default=in_features)

        self.scale = nnx.Param(np.ones(in_features))

        self.out_proj = nnx.Linear(in_features*3, out_features, rngs=rngs)
        
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.out_proj(triact(x * self.scale))

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

