from flax import nnx

from jax import numpy as jnp, tree


class Metric[T](nnx.Variable[T]):
    """
    
    """
    @classmethod
    def pop(cls, module: nnx.Module):
        metrics = nnx.pop(module, cls)
        return loss, metrics


class Loss[T](Metric[T]):
    @classmethod
    def collect(cls, module: nnx.Module):
        losses = nnx.state(module, Loss)

        # Flatten the pytree
        losses = tree.leaves(losses)

        # Sum the flattened elements
        return sum(jnp.sum(loss) for loss in losses)        


class State[T](nnx.Variable[T]): 
    ...
