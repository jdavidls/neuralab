from itertools import pairwise
from typing import List, Tuple

from flax import nnx
from jax import nn
from jax import numpy as jnp
from jax import random


class VAEEncoder(nnx.Module):
    features: List[int]

    def __init__(self, features: List[int], rngs: nnx.Rngs):
        super().__init__()
        self.features = features

        # Crear las capas densas del encoder
        self.layers = []
        for in_features, out_features in pairwise(features[:-1]):
            self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))

        # Capas para generar mu y log_var del espacio latente
        self.mu = nnx.Linear(features[-2], features[-1], rngs=rngs)
        self.log_var = nnx.Linear(features[-2], features[-1], rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Pasar por las capas densas con activación ReLU
        for layer in self.layers:
            x = nn.relu(layer(x))

        # Obtener mu y log_var
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class VAEDecoder(nnx.Module):
    features: List[int]

    def __init__(self, features: List[int], rngs: nnx.Rngs):
        super().__init__()
        self.features = features

        # Crear las capas densas del decoder
        self.layers = []
        for in_features, out_features in pairwise(features[:-1]):
            self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))

        # Capa final para reconstruir la entrada
        self.output_layer = nnx.Linear(features[-2], features[-1], rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # Pasar por las capas densas con activación ReLU
        x = z
        for layer in self.layers:
            x = nn.relu(layer(x))

        # Capa final con activación sigmoid para reconstrucción
        # return jax.nn.sigmoid(self.output_layer(x))
        return self.output_layer(x)


class VAE(nnx.Module):
    features: List[int]
    latent_dim: int
    input_dim: int

    def __init__(self, features: List[int], rngs: nnx.Rngs):
        super().__init__()
        self.features = features
        self.rngs = rngs

        # Inicializar encoder y decoder
        self.encoder = VAEEncoder(features, rngs=rngs)
        self.decoder = VAEDecoder(features[::-1], rngs=rngs)

    @staticmethod
    def reparameterize(
        mu: jnp.ndarray,
        log_var: jnp.ndarray,
        rngs: nnx.Rngs,
    ) -> jnp.ndarray:
        std = jnp.exp(0.5 * log_var)
        eps = random.normal(rngs.vae(), std.shape)
        return mu + eps * std

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Codificar
        mu, log_var = self.encoder(x)

        # Reparametrizar
        z = VAE.reparameterize(mu, log_var, self.rngs)

        # Decodificar
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    @staticmethod
    def kl_loss(mu: jnp.ndarray, log_var: jnp.ndarray) -> jnp.ndarray:
        return -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var))

    @staticmethod
    def bce_loss(x: jnp.ndarray, reconstruction: jnp.ndarray) -> jnp.ndarray:
        return -jnp.sum(
            x * jnp.log(reconstruction + 1e-10)
            + (1 - x) * jnp.log(1 - reconstruction + 1e-10)
        )

    @staticmethod
    def se_loss(x: jnp.ndarray, reconstruction: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(jnp.square(x - reconstruction))

    @staticmethod
    def mse_loss(x: jnp.ndarray, reconstruction: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.square(x - reconstruction))


# # Función de entrenamiento para un paso
# @jax.jit
# def train_step(
#     state: nnx.State, x: jnp.ndarray, rngs: nnx.Rngs
# ) -> Tuple[nnx.State, jnp.ndarray]:
#     def loss_fn(params):
#         state_prime = state.replace(params=params)
#         reconstruction, mu, log_var = state_prime.apply(x, rngs.next())
#         loss = state_prime.module.loss_function(x, reconstruction, mu, log_var)
#         return loss

#     grad = jax.grad(loss_fn)(state.params)
#     state = state.apply_gradients(grad)

#     return state
