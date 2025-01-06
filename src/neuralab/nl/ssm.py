"""Simple, minimal implementation of Mamba in JAX/Flax.

Suggested reading:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    l: sequence length                 (`L` in [1] Algorithm 2)
    f or num_features: feature dim     (`D` in [1] Algorithm 2)
    n or ssm_dim: state space dim      (`N` in [1] Algorithm 2)
    e or expand: expansion factor      (`E` in [1] Section 3.4)
    h or ssm_features: f * expand      (Hidden dimension)
    dt_rank: rank of Δ projection      (See [1] Section 3.6)

State Space Parameters:
    A: state transition matrix         (n, n)
    B: input projection matrix         (n, h)
    C: output projection matrix        (h, n)
    D: skip connection                 (h,)
    Δ: input-dependent step size       (l, h)

Note: B, C are input-dependent (selective) while A, D are static parameters
"""

# %%
from typing import Annotated, Callable

from einops import einsum, rearrange, repeat
from flax import nnx, struct
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Float


def selective_scan(
    u: Float[Array, "l ... h"],
    dt: Float[Array, "l ... h"],
    A: Float[Array, "n n"],
    B: Float[Array, "n h"],
    C: Float[Array, "h n"],
    D: Float[Array, "h"],
    dt_min=0.001,
    dt_max=0.1,
):
    """Selective SSM scan with proper gating mechanism.

    Args:
        u: input sequences             (l, ..., h)
        dt: time delta                 (l, ..., h)
        A: state matrix               (n, n)
        B: input matrix               (n, h)
        C: output matrix              (h, n)
        D: skip connection            (h,)
        dt_min: minimum delta value
        dt_max: maximum delta value

    Returns:
        output: transformed sequence   (l, ..., h)
    """
    batch_shape = u.shape[1:-1]

    # Clamp delta values
    dt = jnp.clip(dt, dt_min, dt_max)

    # Discretize continuous parameters
    ΔA = jnp.exp(einsum(dt, A, "l ... d, d n -> l ... d n"))
    ΔB_u = einsum(dt, B, u, "l ... d, l ... n, l ... d -> l ... d n")

    # Selective scan with state handling
    if False:
        @nnx.scan
        def selective_scan_step(carry, inputs):
            x_t_1 = carry
            ΔA_t, ΔB_u_t, C_t = inputs

            # State update
            x_t = ΔA_t * x_t_1 + ΔB_u_t

            # Output projection
            y_t = einsum(C_t, x_t, "... n, ... d n -> ... d")

            return x_t, y_t

        # Initialize state
        x0 = jnp.zeros(batch_shape + ΔA.shape[-2:])

        # Run scan
        _, y = selective_scan_step(x0, (ΔA, ΔB_u, C))
    else:
        def selective_scan_step(s, c):
            return c[0] * s[0], c[0] * s[1] + c[1]

        # Run scan
        _, x = lax.associative_scan(selective_scan_step, (ΔA, ΔB_u))
        y = einsum(C, x, "... n, ... d n -> ... d")
        

    # Skip connection
    return y + u * D


class SSM(nnx.Module):
    """State Space Model with selective mechanisms."""

    def __init__(
        self,
        features: Annotated[int, "H"],
        state_dim: Annotated[int, "N"],
        dt_rank: Annotated[int, "R"],
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize SSM parameters.

        Args:
            features: Hidden dimension (H)
            state_dim: State space dimension (N)
            dt_rank: Delta projection rank (R)
        """
        self.features = features
        self.state_dim = state_dim
        self.dt_rank = dt_rank

        A = repeat(jnp.arange(1, state_dim + 1), "N -> H N", H=features)
        self.A_log: nnx.Param[Float[Array, "H N"]] = nnx.Param(jnp.log(A))

        self.D: nnx.Param[Float[Array, "H"]] = nnx.Param(jnp.ones(features))

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.ssm_proj = nnx.Linear(
            features, dt_rank + state_dim * 2, use_bias=False, rngs=rngs
        )

        # dt_proj projects Δ from dt_rank to h
        self.dt_proj = nnx.Linear(dt_rank, features, use_bias=True, rngs=rngs)

    def __call__(self, u: Float[Array, "T ... H"]) -> Float[Array, "T ... H"]:
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (l, ... h)    (See Glossary at top for definitions of b, l, h, n...)

        Returns:
            output: shape (b, l, h)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -jnp.exp(self.A_log.value)  # shape (h, n)

        (B, C, dt) = jnp.split(
            self.ssm_proj(u),
            [self.state_dim, 2 * self.state_dim],
            axis=-1,
        )

        Δ = nnx.softplus(self.dt_proj(dt))  # (l, h)

        D = self.D.value

        # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        y = selective_scan(u, Δ, A, B, C, D)

        return y


class Mamba(nnx.Module):
    @struct.dataclass
    class Settings:
        num_features: Annotated[int, "F"]
        expand: Annotated[int, "E"] = 2
        ssm_dim: Annotated[int, "N"] = 16
        kernel_size: int = 4
        use_proj_bias: bool = False
        use_conv_bias: bool = True
        activation: Callable[[Array], Array] = nnx.silu

        @property
        def dt_rank(self):
            return self.num_features // self.ssm_dim

        @property
        def ssm_features(self):
            return int(self.expand * self.num_features)

        @property
        def use_convolution(self):
            return self.kernel_size > 1

        def build(self, rngs: nnx.Rngs):
            return Mamba(self, rngs=rngs)

    def __init__(self, settings: Settings, *, rngs: nnx.Rngs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        self.settings = settings

        self.in_proj = nnx.Linear(
            settings.num_features,
            settings.ssm_features * 2,
            use_bias=settings.use_proj_bias,
            rngs=rngs,
        )

        if settings.use_convolution:
            self.conv = nnx.Conv(
                in_features=settings.ssm_features,
                out_features=settings.ssm_features,
                kernel_size=settings.kernel_size,
                use_bias=settings.use_conv_bias,
                feature_group_count=settings.expand,
                padding="CAUSAL",
                rngs=rngs,
            )

        self.ssm = SSM(
            settings.ssm_features,
            settings.ssm_dim,
            settings.dt_rank,
            rngs=rngs,
        )

        self.out_proj = nnx.Linear(
            settings.ssm_features,
            settings.num_features,
            use_bias=settings.use_proj_bias,
            rngs=rngs,
        )

        self.norm = nnx.RMSNorm(settings.num_features, rngs=rngs)

    def __call__(self, x: Float[Array, "l ... f"]) -> Float[Array, "l ... f"]:
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, h, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        x = self.in_proj(x)
        (x, res) = jnp.split(x, 2, axis=-1)  # shape (l, ..., 2 * f)

        if self.settings.use_convolution:
            x = rearrange(x, 't ... f -> ... t f')
            x = self.conv(x)
            x = rearrange(x, '... t f -> t ... f')

        x = self.settings.activation(x)

        y = self.ssm(x)

        y = y * self.settings.activation(res)

        output = self.out_proj(y)

        return output


if __name__ == "__main__":
    rngs = nnx.Rngs(1)
    ssm = SSM(2, 4, 8, rngs=rngs)
    x = jnp.ones((100, 16, 2))
    y = ssm(x)

    mamba = Mamba(Mamba.Settings(16), rngs=rngs)
    x = jnp.ones((100, 16, 16))
    y = mamba(x)
    

# %%
