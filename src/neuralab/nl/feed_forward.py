from typing import Callable, Optional
from flax import nnx
from jaxtyping import Array, Float
from jax import numpy as jnp
from numpy import pi as PI

from neuralab import fn

class FeedForward(nnx.Module):
    def __init__(
            self, 
            in_features: int, 
            hidden_features: int | float,
            out_features: Optional[int] = None,
            *,
            dropout_rate: float = 0.1,
            activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = nnx.relu,
            residual: bool = True,
            normalization: bool = True,
            rngs: nnx.Rngs
        ):

        self.in_features = in_features

        if isinstance(hidden_features, int):
            self.hidden_features = hidden_features 
        else:
            self.hidden_features = int(in_features * hidden_features)

        if out_features is not None:
            self.out_features = out_features 
        else:
            self.out_features = in_features

        self.activation = activation
        self.residual = residual
        
        self.encoder = nnx.Linear(self.in_features, self.hidden_features, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.decoder = nnx.Linear(self.hidden_features, self.out_features, rngs=rngs)

        if normalization:
            self.norm = nnx.LayerNorm(self.out_features, rngs=rngs)


    def __call__(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.residual:
            x = x + x
        if hasattr(self, "norm"):
            x = self.norm(x)
        return x

class TriactForward(nnx.Module):
    def __init__(self, input_features: int, output_features: int, rngs: nnx.Rngs):
        self.input_features = input_features
        self.output_features = output_features

        self.scaler = nnx.Param(jnp.ones(input_features) * PI)
        self.proj = nnx.Linear(3 * input_features, output_features, rngs=rngs)

    def __call__(self, x):
        return self.proj(fn.triact(x * self.scaler))
