from flax import nnx

class Loss[T](nnx.Variable[T]): 
    ...

class State[T](nnx.Variable[T]): 
    ...
