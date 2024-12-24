# %%
from typing import Optional, Tuple

import jax
from einops import einsum, rearrange, repeat
from flax import nnx
from jax import lax
from jax import numpy as np
from jax import random
from jaxtyping import Array, Float

from neuralab.nl.common import State


def make_hippo(N: int) -> Tuple[Float[Array, "N N"], Float[Array, "N 1"]]:
    n = np.arange(N)
    P = np.sqrt(1 + 2 * n)
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(n)
    B = np.sqrt(2 * n + 1.0)
    B = B[:, None]
    return -A, B


def discretize(
    A: Float[Array, "N N"],
    B: Float[Array, "N 1"],
    dt: Float[Array, ""] | float,
):
    I = np.eye(A.shape[0])
    dt2A = dt / 2 * A
    BL = np.linalg.inv(I - dt2A)
    Ab = BL @ (I + dt2A)
    Bb = (BL * dt) @ B
    return Ab, Bb


def hippo_scan(
    u: Float[Array, "L ..."],
    Ab: Float[Array, "N N"],
    Bb: Float[Array, "N 1"],
    state: Optional[Float[Array, "... N"]] = None,
) -> Tuple[
    Float[Array, "... N"],
    Float[Array, "L ... N"],
]:

    N = Ab.shape[0]

    if state is None:
        state = np.zeros((N,))

    #print(u.shape, Ab.shape, Bb.shape, state.shape)

    Bb_u = einsum(Bb, u[..., None], "N one, L ... one -> L ... N")

    def hippo_step(x0, Bb_u_k):
        x1 = einsum(Ab, x0, "N M, ... M -> ... N") + Bb_u_k
        return x1, x1

    return lax.scan(hippo_step, state, Bb_u)


def hippo_encode(u, N: int, dt: Optional[float]=None):
    A, B = make_hippo(N)
    if dt is None:
        dt = 1 / len(u)
    Ab, Bb = discretize(A, B, dt)
    _, x = hippo_scan(u, Ab, Bb)
    return x


def eval_legendre(n: int, x: np.ndarray) -> np.ndarray:
    """
    Evaluates Legendre polynomials of specified degrees at provided points.

    Args:
        n int: Degrees for which Legendre polynomials are to be evaluated.
        x (jnp.ndarray): Points at which to evaluate the Legendre polynomials.

    Returns:
        jnp.ndarray: Array of Legendre polynomial values.

    """
    # Implementation
    # --------------
    #     Set the 0th degree Legendre polynomial to 1 and the 1st degree Legendre
    #     polynomial to x.

    p_0 = np.ones_like(x, dtype=np.float32)
    p_1 = x

    #     Compute the Legendre polynomials for degrees 2 to max_n using the recurrence
    #     relation:
    #         P_{n+1}(x) = ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1)

    def recurrence(p, n):
        p_n_minus_1, p_n = p
        p_n_plus_1 = ((2 * n + 1) * x * p_n - n * p_n_minus_1) / (n + 1)

        return (p_n, p_n_plus_1), p_n_plus_1

    _, p_n = lax.scan(recurrence, (p_0, p_1), np.arange(2, n))

    #     Concatenate the 0th and 1st degree Legendre polynomials with the computed

    return np.concatenate([p_0[None], p_1[None], p_n], axis=0)


def hippo_basis(N: int, L: float, dt: float) -> Float[Array, "L N"]:
    n = np.arange(N)
    l = np.arange(L)
    t = np.exp(-l * dt)
    basis = eval_legendre(N, 1 - 2 * t).T
    basis = basis * (2 * n + 1) ** 0.5 * (-1) ** n
    return basis


def hippo_decode(
    x: Float[Array, "L ... N"], 
    dt: Optional[float] = None
) -> Float[Array, "L"]:
    L, *_, N = x.shape
    if dt is None:
        dt = 1 / L
    basis = hippo_basis(N, L, dt)
    return einsum(basis, x, "L N, ... N -> ... L")


def init_log_dt(dt_min=0.001, dt_max=0.1):
    def init(key):
        return random.uniform(key, (1,)) * (np.log(dt_max) - np.log(dt_min)) + np.log(
            dt_min
        )

    return init


def log_dt_initializer(hippos: int, *, dt_min=0.001, dt_max=0.1, rngs: nnx.Rngs):
    return random.uniform(rngs.next(), (hippos,)) * (
        np.log(dt_max) - np.log(dt_min)
    ) + np.log(dt_min)


class HiPPOEncoder(nnx.Module):

    def __init__(
        self,
        out_features: int,
        hippos: int,
        learn_transition: bool = False,
        dt_min=0.001,
        dt_max=0.1,
        *,
        rngs: nnx.Rngs
    ):
        self.hippos = hippos
        self.out_features = out_features

        A, B = make_hippo(out_features)
        A = repeat(A, "N M -> H N M ", H=hippos)
        B = repeat(B, "N 1 -> H N 1 ", H=hippos)

        self.A = nnx.Param(A) if learn_transition else nnx.Cache(A)
        self.B = nnx.Param(B) if learn_transition else nnx.Cache(B)

        # self.log_dt = nnx.Param(
        #     jax.random.uniform(rngs.next(), (hippos,))
        #     * (np.log(dt_max) - np.log(dt_min))
        #     + np.log(dt_min)
        # )

        self.log_dt = nnx.Param(np.linspace(dt_min, dt_max, hippos))

        self.state: State[Optional[Float[Array, "H N"]]] = State(None)

    @property
    def dt(self):
        return self.log_dt.value
        # return nn.sigmoid(self.log_dt.value)

    @nnx.jit
    def __call__(self, u: Float[Array, "L "]) -> Float[Array, "L H N"]:
        Ab, Bb = jax.vmap(discretize, in_axes=(0, 0, 0), out_axes=(0, 0))(self.A.value, self.B.value, self.dt)

        state, x = jax.vmap(hippo_scan, in_axes=(None, 0, 0, 0), out_axes=(0, 1))(
            u, Ab, Bb, self.state.value
        )

        self.state.value = state

        return x


class HiPPODecoder(nnx.Module):

    def __init__(self, in_features: int, length: int):
        self.basis = nnx.Cache(hippo_basis(in_features, length, 1 / length))

    def __call__(self, x: Float[Array, "... N"]) -> Float[Array, "... L"]:
        return einsum(self.basis.value, x, "L N, ... N -> ... L")


if __name__ == "__main__" and False:
    import matplotlib.pyplot as plt

    rngs = nnx.Rngs(3)
    basis = eval_legendre(32, np.linspace(-1, 1, 201))
    plt.pcolormesh(basis)
    plt.colorbar()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rngs = nnx.Rngs(3)
    N, L, C = 32, 512, 2
    u = np.cumsum(jax.random.normal(rngs.next(), (L,)), axis=0)

    enc = HiPPOEncoder(N, 4, dt_min=1 / L, rngs=rngs)

    dec = HiPPODecoder(N, L)

    x = enc(u)

    plt.pcolormesh(rearrange(x, "l h n -> l (h n)"))
    plt.colorbar()
    plt.show()

    OFFSET, H = L - 1, 0
    v = dec(x[OFFSET, H])

    plt.plot(u)
    plt.plot(np.flip(v)[-OFFSET:])
    plt.show()
