# VAE doble donde se entrenan dos autoencoders en una ventana de tiempo 
# partida por la mitad (el sample de referencia que representa el momento presente)

import jax
from jax import numpy as jnp, nn
from jax import random, grad, jit, vmap
from flax import nnx
import optax


class Activation(nnx.Module):
    def __init__(self, activation):
        self.activation = activation
    
    def __call__(self, x):
        return self.activation(x)


class VAE(nnx.Module):
    def __inig__(
            self, 
            features: list[int], 
            rngs: nnx.Rngs
        ):

        self.encoder = nnx.Sequential(
            nnx.Linear(features[0], rngs=rngs),
            Activation(nn.relu),
        )

        self.encoder1_hidden = nnx.Linear(self.hidden_dim)
        self.mu1_layer = nnx.Linear(self.latent_dim)
        self.logvar1_layer = nnx.Linear(self.latent_dim)


class TemporalVAE(nnx.Module):
    hidden_dim: int
    latent_dim: int
    window_size: int  # Tamaño total de la ventana temporal
    
    def __init__(self):
        # Cada ventana será de tamaño window_size/2
        #self.half_window = self.window_size // 2
        
        # VAE para la primera mitad de la ventana (pasado)
        self.encoder1_hidden = nnx.Linear(self.hidden_dim)
        self.mu1_layer = nnx.Linear(self.latent_dim)
        self.logvar1_layer = nnx.Linear(self.latent_dim)
        self.decoder1_hidden = nnx.Linear(self.hidden_dim)
        self.output1_layer = nnx.Linear(self.half_window)
        
        # VAE para la segunda mitad (presente)
        self.encoder2_hidden = nnx.Linear(self.hidden_dim)
        self.mu2_layer = nnx.Linear(self.latent_dim)
        self.logvar2_layer = nnx.Linear(self.latent_dim)
        self.decoder2_hidden = nnx.Linear(self.hidden_dim)
        self.output2_layer = nnx.Linear(self.half_window)
    
    def encode_past(self, x):
        x = nnx.relu(self.encoder1_hidden(x))
        mu = self.mu1_layer(x)
        logvar = self.logvar1_layer(x)
        return mu, logvar
    
    def encode_present(self, x):
        x = nnx.relu(self.encoder2_hidden(x))
        mu = self.mu2_layer(x)
        logvar = self.logvar2_layer(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, mu.shape)
        return mu + eps * std
    
    def decode_past(self, z):
        z = nnx.relu(self.decoder1_hidden(z))
        return nnx.sigmoid(self.output1_layer(z))
    
    def decode_present(self, z):
        z = nnx.relu(self.decoder2_hidden(z))
        return nnx.sigmoid(self.output2_layer(z))
    
    def __call__(self, x, rng):
        # Dividir la entrada en dos mitades
        x_past = x[:, :self.half_window]
        x_present = x[:, self.half_window:]
        
        # Procesar primera mitad (pasado)
        rng1, rng2 = random.split(rng)
        mu1, logvar1 = self.encode_past(x_past)
        z1 = self.reparameterize(mu1, logvar1, rng1)
        recon_past = self.decode_past(z1)
        
        # Procesar segunda mitad (presente)
        mu2, logvar2 = self.encode_present(x_present)
        z2 = self.reparameterize(mu2, logvar2, rng2)
        recon_present = self.decode_present(z2)
        
        return (recon_past, recon_present), (mu1, logvar1, z1), (mu2, logvar2, z2)

def cosine_distance(v1, v2):
    """Calcula la distancia del coseno entre dos vectores."""
    norm1 = jnp.sqrt(jnp.sum(v1 * v1))
    norm2 = jnp.sqrt(jnp.sum(v2 * v2))
    dot_product = jnp.sum(v1 * v2)
    return 1.0 - dot_product / (norm1 * norm2)

def temporal_vae_loss(params, model, batch, rng, lambda_kl=1.0, lambda_cosine=0.5):
    """
    Función de pérdida combinada:
    1. Reconstrucción para ambos VAEs
    2. KL divergence para ambos VAEs
    3. Distancia coseno entre espacios latentes
    """
    def loss_fn(x):
        (recon_past, recon_present), (mu1, logvar1, z1), (mu2, logvar2, z2) = \
            model.apply({'params': params}, x, rng)
        
        # Dividir datos originales
        x_past = x[:, :model.half_window]
        x_present = x[:, model.half_window:]
        
        # Pérdida de reconstrucción para ambos VAEs
        recon_loss_past = -jnp.sum(
            x_past * jnp.log(recon_past + 1e-8) + 
            (1 - x_past) * jnp.log(1 - recon_past + 1e-8)
        )
        recon_loss_present = -jnp.sum(
            x_present * jnp.log(recon_present + 1e-8) + 
            (1 - x_present) * jnp.log(1 - recon_present + 1e-8)
        )
        
        # KL divergence para ambos VAEs
        kl_loss_past = -0.5 * jnp.sum(1 + logvar1 - jnp.square(mu1) - jnp.exp(logvar1))
        kl_loss_present = -0.5 * jnp.sum(1 + logvar2 - jnp.square(mu2) - jnp.exp(logvar2))
        
        # Distancia coseno entre espacios latentes
        cosine_loss = cosine_distance(z1, z2)
        
        # Pérdida total
        total_loss = (recon_loss_past + recon_loss_present) + \
                    lambda_kl * (kl_loss_past + kl_loss_present) + \
                    lambda_cosine * cosine_loss
        
        return total_loss
    
    return jnp.mean(vmap(loss_fn)(batch))

@jit
def train_step(state, batch, rng):
    loss_fn = lambda params: temporal_vae_loss(params, state.model, batch, rng)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, loss_fn(state.params)

def create_temporal_vae(rng, window_size, learning_rate=1e-3):
    hidden_dim = 512
    latent_dim = 32
    
    model = TemporalVAE(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        window_size=window_size
    )
    
    # Inicializar con una entrada de ejemplo
    params = model.init(rng, jnp.ones((1, window_size)), rng)
    
    tx = optax.adam(learning_rate)
    state = optax.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    
    return state, model

# Función para generar nuevas muestras
def generate_samples(model, params, rng, num_samples=1):
    """Genera muestras para ambas mitades de la ventana temporal."""
    rng1, rng2 = random.split(rng)
    
    # Generar latentes
    z1 = random.normal(rng1, (num_samples, model.latent_dim))
    z2 = random.normal(rng2, (num_samples, model.latent_dim))
    
    # Decodificar
    past_samples = model.apply({'params': params}, None, method=model.decode_past)(z1)
    present_samples = model.apply({'params': params}, None, method=model.decode_present)(z2)
    
    return past_samples, present_samples

#%%

if __name__ == '__main__':
    

    ...
