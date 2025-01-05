from typing import Callable, Optional
from flax import nnx
from jaxtyping import Array, Float

class FeedForward(nnx.Module):
    def __init__(
            self, 
            in_features: int, 
            *,
            expand: int = 2, 
            dropout_rate: float = 0.1,
            out_features: Optional[int] = None, 
            activation: Callable[[Float[Array, "..."]], Float[Array, "..."]] = nnx.relu,
            rngs: nnx.Rngs
        ):

        self.in_features = in_features
        self.out_features = out_features = optional(out_features, default=in_features)
        self.activation = activation
        self.depth = depth = in_features * expand
        
        self.encoder = nnx.Linear(in_features, depth, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.decoder = nnx.Linear(depth, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x


# nnx.LayerNorm with groups...