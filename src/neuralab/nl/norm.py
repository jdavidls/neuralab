from flax import nnx
from jax import numpy as jnp, lax
from jaxtyping import Array, Float


class RMSNorm(nnx.Module):
    """Root Mean Square Normalization layer.
    
    Implements RMS normalization as described in https://arxiv.org/abs/1910.07467
    The normalization is performed over the last dimension.
    
    Attributes:
        num_features: Number of input features to normalize over
        weights: Learnable scale parameters
    """

    def __init__(
        self,
        num_features: int,
    ):
        """Initialize RMSNorm layer.
        
        Args:
            num_features: Number of features in the input tensor's last dimension
        """
        self.num_features = num_features
        self.weights = nnx.Param(jnp.ones(num_features))

    def __call__(self, x: Array, eps: float = 1e-5) -> Array:
        """Apply RMS normalization.
        
        Args:
            x: Input tensor to normalize
            eps: Small constant for numerical stability
            
        Returns:
            Normalized tensor with the same shape as input
        """
        if x.shape[-1] != self.num_features:
            raise ValueError(f"Expected last dimension to be {self.num_features}, got {x.shape[-1]}")
            
        scale = lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
        return x * scale * self.weights