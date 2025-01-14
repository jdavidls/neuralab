from flax import nnx

from jax import numpy as jnp, tree


class Metric[T](nnx.Variable[T]):
    ...
   

class Loss[T](Metric[T]):
    ...


class State[T](nnx.Variable[T]): 
    ...
