# %% Gaussian Self Attention

from typing import Tuple

from flax import nnx
from jax import lax, nn
from jax import numpy as jnp
from jax import random
from jax import scipy as sp
from jax.scipy import linalg
from jaxtyping import Array, Float
from jax.scipy.special import logsumexp


from neuralab.nl.common import Loss

LOG_TAU = jnp.log(2 * jnp.pi)


import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.special import logsumexp

def log_emission_prob(obs, mu, cov):
    """
    Calcula la probabilidad logarítmica de una observación multivariante o_t 
    bajo una distribución gaussiana multivariante N(mu, cov).

    Args:
        o_t (jax.numpy.ndarray): Vector de observaciones de tamaño (d,).
        mu (jax.numpy.ndarray): Vector de medias de tamaño (d,).
        cov (jax.numpy.ndarray): Matriz de covarianza (d, d).

    Returns:
        log_prob (float): Logaritmo de la probabilidad de emisión.
    """
    d = mu.shape[0]
    cov_inv = jnp.linalg.inv(cov)
    cov_det = jnp.linalg.det(cov)
    
    # Término cuadrático (o_t - mu)^T cov^-1 (o_t - mu)
    diff = obs - mu
    quadratic_term = -0.5 * jnp.dot(diff.T, jnp.dot(cov_inv, diff))
    
    # Término normalizador: log((2π)^(d/2) * |Σ|^(1/2))
    log_normalizer = 0.5 * d * jnp.log(2 * jnp.pi) + 0.5 * jnp.log(cov_det)
    
    # Log-probabilidad
    log_prob = quadratic_term - log_normalizer
    return log_prob


class MultivariateGaussian(nnx.Module):
    def __init__(self, num_states: int, num_observations: int, rngs: nnx.Rngs):
        self.num_states = num_states
        self.num_features = num_observations

        # Parámetros de las medias de las distribuciones gaussianas
        self.means = nnx.Param(random.normal(rngs.params(), (num_states, num_observations)))

        # Parámetros de las covarianzas de las gaussianas
        self.covs = nnx.Param(jnp.eye(num_observations, num_observations))

    def __call__(self, observations):
        return log_emission_prob(observations, self.means, self.covs)


def log_forward(log_A, log_probs):
    """
    Algoritmo Forward en espacio logarítmico, utilizando log(A) y lax.scan.

    Args:
        log_A (jax.numpy.ndarray): Matriz de transición en logaritmos (num_states, num_states).
        log_probs (jax.numpy.ndarray): Probabilidades de emisión en espacio logarítmico (T, num_states).

    Returns:
        log_alpha: Probabilidad logarítmica total de la secuencia.
    """
    T = log_probs.shape[0]

    def step(log_alpha_prev, log_prob_t):
        """
        Propagación hacia adelante en un paso de tiempo.
        """
        # Calculamos log_alpha_t(j) = logsumexp(log_alpha_prev + log_A[:, j]) + log_prob_t[j]
        log_alpha_t = logsumexp(log_alpha_prev[:, None] + log_A, axis=0) + log_prob_t
        return log_alpha_t, log_alpha_t

    # Inicializamos log_alpha[0] con las probabilidades de emisión en t=0
    log_alpha_0 = log_probs[0]

    # Aplicamos lax.scan para propagar hacia adelante
    # _, log_alpha_seq = lax.scan(step, log_alpha_0, log_probs[1:])
    _, log_alpha = lax.scan(step, log_alpha_0, log_probs)

    # Calculamos la probabilidad logarítmica total de la secuencia
    # verosimilitud
    log_likelihood = logsumexp(log_alpha[-1], axis=-1)

    return log_alpha, log_likelihood


class GaussianHMM(nnx.Module):
    def __init__(self, num_states: int, num_features: int, rngs: nnx.Rngs):
        self.num_states = num_states
        self.num_features = num_features

        # Multivariate Gaussian 
        self.multivariate_gaussian = MultivariateGaussian(num_states, num_features, rngs)

        # Transition parameters  
        self.log_transition_logits = nnx.Param(
            jnp.zeros((self.num_states, self.num_states))
        )

    @property
    def log_transition_matrix(self):
        return self.log_transition_logits - logsumexp(self.log_transition_logits.value, axis=-1, keepdims=True)

    def __call__(self, obs, time_scaling=1.0):
        log_A = self.log_transition_matrix * time_scaling
        log_probs = self.multivariate_gaussian(obs)
        log_alpha, log_likelihood = log_forward(log_A, log_probs)
        return log_alpha, log_likelihood


    @classmethod
    def stack(cls, num_layers, *args, **kwargs):
        def layer(_):
            return cls(*args, **kwargs)
        
        return nnx.vmap(layer)(jnp.arange(num_layers))


if __name__ == "__main__":
    import optax
    from matplotlib import pyplot as plt
    
    # Generate synthetic data
    def generate_synthetic_data(key, num_sequences=20, seq_length=100):
        # Define true parameters for 3 states
        true_means = jnp.array([
            [-2.0, 0.0],  # State 1
            [2.0, 0.0],   # State 2
            [0.0, 2.0],   # State 3
        ])
        
        true_covs = jnp.array([
            [[0.5, 0.0], [0.0, 0.5]],  # State 1
            [[0.5, 0.0], [0.0, 0.5]],  # State 2
            [[0.5, 0.0], [0.0, 0.5]],  # State 3
        ])
        
        # True transition matrix
        true_A = jnp.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        
        sequences = []
        states = []
        
        for i in range(num_sequences):
            key, subkey = random.split(key)
            # Initial state
            state = random.categorical(subkey, jnp.log(jnp.ones(3)/3.))
            seq = []
            state_seq = []
            
            for t in range(seq_length):
                key, subkey1, subkey2 = random.split(key, 3)
                # Generate observation
                mean = true_means[state]
                cov = true_covs[state]
                obs = random.multivariate_normal(subkey1, mean, cov)
                seq.append(obs)
                state_seq.append(state)
                
                # Transition
                state = random.categorical(subkey2, jnp.log(true_A[state]))
            
            sequences.append(jnp.stack(seq))
            states.append(jnp.array(state_seq))
            
        return jnp.stack(sequences), jnp.stack(states)

    @nnx.jit
    def train_step(model, optimizer, sequences):
        
        def loss_fn(model):

            @nnx.vmap
            def batch_loss(sequence):
                model(sequence)
                return Loss.collect(model)

            losses = batch_loss(sequences)  # [B]
            return jnp.mean(losses)
            
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss
            
    # Training loop
    def train_hmm(model, optimizer, sequences, true_states, num_epochs=10000):

        for epoch in range(num_epochs):
            model.train(training=True)
            loss = train_step(model, optimizer, sequences)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if epoch % 100 == 0:
                eval_hmm(model, sequences[0], true_states[0])

    def eval_hmm(model, test_sequence, true_states):
        model.eval(training=False)
        predicted_states = model(test_sequence)
        
        # Convert true states to one-hot encoding
        true_states_onehot = nn.one_hot(true_states, num_states)  # [T,S]
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(121)
        plt.title("True States")
        plt.imshow(true_states_onehot.T, aspect='auto', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.colorbar(label='State Probability')
        
        plt.subplot(122)
        plt.title("Predicted States")
        plt.imshow(predicted_states.T, aspect='auto', cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.colorbar(label='State Probability')
        
        plt.tight_layout()
        plt.show()


    # Main execution
    key = random.PRNGKey(24)
    sequences, true_states = generate_synthetic_data(key)
    
    num_states = 3
    num_features = 2
    rngs = nnx.Rngs(0)
    
    model = MultivariateGaussianEmiter(num_states, num_features, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))


    #%%
    # Train
    train_hmm(model, optimizer, sequences, true_states, num_epochs=2000)
    nnx.display(model)

    #%%
