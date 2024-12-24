from itertools import pairwise
import jax
from jax import numpy as jnp, random
from flax import nnx
from typing import List, Tuple


class VAEEncoder(nnx.Module):
    features: List[int]
    latent_dim: int

    def __init__(self, features: List[int], latent_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.features = features
        self.latent_dim = latent_dim

        # Crear las capas densas del encoder
        self.layers = []
        for in_features, out_features in pairwise(features):
            self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))

        # Capas para generar mu y log_var del espacio latente
        self.mu = nnx.Linear(features[-1], latent_dim, rngs=rngs)
        self.log_var = nnx.Linear(features[-1], latent_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Pasar por las capas densas con activación ReLU
        for layer in self.layers:
            x = jax.nn.relu(layer(x))

        # Obtener mu y log_var
        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class VAEDecoder(nnx.Module):
    features: List[int]
    output_dim: int

    def __init__(self, features: List[int], output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.features = features
        self.output_dim = output_dim

        # Crear las capas densas del decoder
        self.layers = []
        for in_features, out_features in pairwise(features):
            self.layers.append(nnx.Linear(in_features, out_features, rngs=rngs))

        # Capa final para reconstruir la entrada
        self.output_layer = nnx.Linear(features[-1],  output_dim, rngs=rngs)

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # Pasar por las capas densas con activación ReLU
        x = z
        for layer in self.layers:
            x = jax.nn.relu(layer(x))

        # Capa final con activación sigmoid para reconstrucción
        return jax.nn.sigmoid(self.output_layer(x))


class VAE(nnx.Module):
    features: List[int]
    latent_dim: int
    input_dim: int

    def __init__(self, features: List[int], latent_dim: int, input_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.features = features
        self.latent_dim = latent_dim
        self.input_dim = input_dim


        # Inicializar encoder y decoder
        self.encoder = VAEEncoder(features, latent_dim, rngs=rngs)
        self.decoder = VAEDecoder(features[::-1], input_dim, rngs=rngs)

    def reparameterize(
        self, mu: jnp.ndarray, log_var: jnp.ndarray, key: nnx.RngStream
    ) -> jnp.ndarray:
        std = jnp.exp(0.5 * log_var)
        eps = random.normal(key(), std.shape)
        return mu + eps * std

    def __call__(
        self, x: jnp.ndarray, key: nnx.RngStream
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Codificar
        mu, log_var = self.encoder(x)

        # Reparametrizar
        z = self.reparameterize(mu, log_var, key)

        # Decodificar
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var


    def loss_function(
        self,
        x: jnp.ndarray,
        reconstruction: jnp.ndarray,
        mu: jnp.ndarray,
        log_var: jnp.ndarray,
    ) -> jnp.ndarray:
        # Pérdida de reconstrucción (BCE)
        recon_loss = -jnp.sum(
            x * jnp.log(reconstruction + 1e-10)
            + (1 - x) * jnp.log(1 - reconstruction + 1e-10)
        )

        # Pérdida KL
        kl_loss = -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var))

        return recon_loss + kl_loss


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

