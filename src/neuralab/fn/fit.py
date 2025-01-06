from flax import nnx

class Model(nnx.Module):
    ...


class Fit[M: Model]:
    '''
    Fit es un servicio que entrena un modelo en segundo plano.
    permitiendo al programa continuar con el flujo de de trabajo.

    a traves de la estancia de este objeto el programa puede interactuar 
    con el flujo de entrenamiento (mediante el uso de bservables).
    
    '''
    def __init__(self, model: M):
        self.model = model

    def __call__(self, *args, **kwargs):
        ...


def fit[M: Model](model: M) -> Fit[M]:
    """
    Transforma el modelo es un objeto entrenable
    """
    return Fit(model)
    