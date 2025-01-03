from flax import nnx

from jax import tree_util, numpy as jnp


class Metric[T](nnx.Variable[T]):
    """
    
    """
    ...
class Loss[T](Metric[T]):
    @classmethod
    def collect(cls, module: nnx.Module):
        losses = nnx.pop(module, cls)

        # Flatten the pytree
        losses = tree_util.tree_leaves(losses)

        # Sum the flattened elements
        return sum(jnp.sum(loss) for loss in losses)        



class State[T](nnx.Variable[T]): 
    ...
