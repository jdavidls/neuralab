from flax import nnx

from jax import numpy as jnp, tree


class Metric[T](nnx.Variable[T]):
    ...
   

class Loss[T](Metric[T]):
    ...


class State[T](nnx.Variable[T]): 
    ...

def reset_state(root: nnx.Module):
    for path, module in nnx.iter_graph(root):
        if isinstance(module, nnx.Module):
            for name, value in vars(module).items():
                if isinstance(value, State):
                    setattr(module, name, State(None))

