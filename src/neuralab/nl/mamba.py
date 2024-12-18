"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""

from typing import Annotated, Optional, TypeVar
from typing_extensions import Doc
import jax
import jax.numpy as np
from flax import nnx
from einops import rearrange, repeat, einsum
from jaxtyping import Float, Array

from neuralab.nn.hippo import make_hippo


T = TypeVar('T')
def optional(value: Optional[T], default: T) -> T:
    return default if value is None else value

class MambaMixer(nnx.Module):
    def __init__(
            self, 
            d_model:int, 
            expand: Annotated[int, Doc("expansion factor, defaults to 2")] = 2,
            d_state: Annotated[
                int, 
                Doc("size of the state space's state, defaults to 16")
            ] = 16, 
            *,
            d_conv:int = 4,
            d_inner:Annotated[Optional[int], Doc("inner hidden dim, defaults to expand * d_model")]=None,
            dt_rank: Annotated[Optional[int], Doc("rank of Δ, defaults to d_model / d_state")] = None,
            use_proj_bias: bool = False,
            use_conv_bias: bool = True, 
            rngs: nnx.Rngs
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        self.d_state = d_state = optional(d_state, 16)
        self.d_inner = d_inner = optional(d_inner, expand * d_model)
        self.dt_rank = dt_rank = optional(dt_rank, d_model // d_state)

        self.in_proj = nnx.Linear(
            d_model, 
            d_inner * 2, 
            use_bias=use_proj_bias, 
            rngs=rngs
        )

        self.conv1d = nnx.Conv(
            in_features=d_inner,
            out_features=d_inner,
            kernel_size=d_conv,
            use_bias=use_conv_bias,
            feature_group_count=expand,
            padding="CAUSAL", 
            rngs=rngs
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nnx.Linear(
            d_inner, 
            dt_rank + d_state * 2, 
            use_bias=False, 
            rngs=rngs
        )
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nnx.Linear(
            dt_rank, 
            d_inner, 
            use_bias=True, 
            rngs=rngs
        )

        #A = repeat(np.arange(1, d_state + 1), 'n -> d_in n', d_in=d_inner)
        A, _ = make_hippo(d_state)[0]
        A = repeat(np.diagonal(A), 'n -> d_in n', d_in=d_inner)

        self.A_log = nnx.Param(np.log(A))

        self.D = nnx.Param(np.ones(d_inner))

        self.out_proj = nnx.Linear(
            d_inner, 
            d_model, 
            use_bias=use_proj_bias, 
            rngs=rngs
        )
        

    def __call__(self, x: Float[Array, "... l d"]) -> Float[Array, "... l d"]:
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        # (b, l, d) = x.shape
        
        (x, res) = np.split(
            self.in_proj(x),  # shape (b, l, 2 * d_in)
            2,
            axis=-1
        )

        #x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)#[:, :, :l]
        #x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = nnx.silu(x)

        y = self.ssm(x)
        
        y = y * nnx.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -np.exp(self.A_log.float())  # shape (d_in, n)

        (Δ, B, C) = rearrange(self.x_proj(x), "l (dt_rank n m) -> l dt_rank, l n 1, l 1 m", dt_rank=self.dt_rank, n=n, m=n)

        Δ = nnx.softplus(self.dt_proj(Δ))  # (l, d_in)
        
        D = self.D.float()
        
        y = selective_scan(x, Δ, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
def selective_scan(
        u: Float[Array, "l d_in"],
        delta: Float[Array, "l d_in"],
        A: Float[Array, "d_in n"], 
        B: Float[Array, "l n 1"], 
        C: Float[Array, "l 1 n"], 
        D: Float[Array, "d_in"]
):
    """Does selective scan algorithm. See:
        - Section 2 State Space Models in the Mamba paper [1]
        - Algorithm 2 in Section 3.2 in the Mamba paper [1]
        - run_SSM(A, B, C, u) in The Annotated S4 [2]

    This is the classic discrete state space formula:
        x(t + 1) = Ax(t) + Bu(t)
        y(t)     = Cx(t) + Du(t)
    except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

    Args:
        u: shape (l, d_in)    
        delta: shape (l, d_in)
        A: shape (d_in, n)
        B: shape (l, n, 1)
        C: shape (l, 1, n)
        D: shape (d_in,)
        
    Returns:
        output: shape (l, d_in)

    Official Implementation:
        selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
        Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
        
    """
    #(d_in, n) = A.shape
    
    # Discretize continuous parameters (A, B)
    # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
    # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
    #   "A is the more important term and the performance doesn't change much with the simplification on B"
    ΔA = np.exp(einsum(delta, A, 'l d_in, d_in n -> l n d_in'))
    ΔB_u = einsum(delta, B, u, 'l d_in, l n 1, l d_in -> l d_in n')
    
    # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
    def step(x0, _): 
        ΔA, ΔB_u, ΔC = _
        x = ΔA @ x0 + ΔB_u
        y = ΔC @ x
        return x, y
    
    x0 = np.zeros_like(A) # (d_in, n)
    _, y = jax.lax.scan(step, x0, (ΔA, ΔB_u, C))
    
    y = y + u * D

    return y

