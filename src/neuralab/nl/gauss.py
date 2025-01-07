# %%

from typing import Tuple

from flax import nnx
from jax import lax, nn
from jax import numpy as jnp
from jax import random
from jax import scipy as sp
from jax.scipy import linalg
from jaxtyping import Array, Float

from neuralab.nl.common import Loss

LOG_TAU = jnp.log(2 * jnp.pi)

# Gaussian Attention

def log_emission_prob(
    obs: Float[Array, "D"], 
    mean: Float[Array, "K D"], 
    covs: Float[Array, "K D D"]
) -> Float[Array, "K"]:
    """Calculate log emission probability for multivariate Gaussian.
    
    Args:
        obs: Observation vector of dimension D
        mean: Mean vectors for K states, each of dimension D
        covs: Covariance matrices for K states, each DxD
        
    Returns:
        Log probability of observation under Gaussian distribution for each state
    """
    D = obs.shape[0]  # dimension
    diff = obs - mean  # [K, D]
    
    # Add small diagonal term for numerical stability
    covs_stable = covs + 1e-6 * jnp.eye(covs.shape[-1])
    
    # Compute Cholesky decomposition: Σ = LL^T
    L = linalg.cholesky(covs_stable)  # [K, D, D]
    
    # Solve L^T y = diff for y using back substitution
    solve_L = linalg.solve_triangular(L, diff[..., None], lower=True)[..., 0]  # [K, D]
    
    # log|Σ| = 2 * sum(log(diag(L)))
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)  # [K]
    
    # Final log probability
    return -0.5 * (D * LOG_TAU + log_det + jnp.sum(solve_L**2, axis=-1))


def viterbi(
    log_A: Float[Array, "S S"],
    log_pi: Float[Array, "S"],
    log_emissions: Float[Array, "T S"],
    temperature: float = 1.0
) -> Tuple[Float[Array, "T S"], Float[Array, "T S"]]:
    """Viterbi algorithm for computing most likely state sequence.
    
    Mathematical formulation:
    
    Forward pass (log-space):
    α[t,j] = log P(o₁,...,oₕ, q[t]=j | λ)
    α[t,j] = log(Σᵢ exp(α[t-1,i] + log A[i,j])) + log B[j,oₜ]
    
    With temperature T:
    α[t,j] = T * log(Σᵢ exp((α[t-1,i] + log A[i,j])/T)) + log B[j,oₜ]
    
    Backward pass (backtracking):
    q*[t] = argmax_j α[t,j]
    
    Parameters:
    -----------
    log_A : array (n_states, n_states)
        Log transition matrix, log P(q[t+1]=j|q[t]=i)
    log_pi : array (n_states,)
        Log initial state distribution
    log_emissions : array (seq_len, n_states)
        Log emission probabilities for each observation
    temperature : float
        Temperature parameter for soft Viterbi (T=1 for standard Viterbi)
        Higher T → more smoothing of probabilities
    
    Returns:
    --------
    soft_path : array (seq_len, n_states)
        Soft state assignments P(q[t]=j|o₁,...,o_T)
    loss : float
        Negative log likelihood of most probable path
    """
    def step(carry_n_1, log_emission_n):
        log_alpha_n_1, log_delta_n_1 = carry_n_1

        # Forward pass
        log_alpha_n = sp.special.logsumexp(
            log_alpha_n_1[:, None] + log_A, 
            axis=0
        ) + log_emission_n

        # Soft Viterbi
        v_tmp = log_delta_n_1[:, None] + log_A
        log_probs = v_tmp / temperature
        weights = nn.softmax(log_probs, axis=0)
        log_delta_n = jnp.sum(weights * v_tmp, axis=0) + log_emission_n

        # Normalization
        #log_alpha_n = log_alpha_n - jnp.max(log_alpha_n)
        #log_delta_n = log_delta_n - jnp.max(log_delta_n)

        return (log_alpha_n, log_delta_n), (log_alpha_n, log_delta_n)

    # Initialize
    log_alpha_0 = log_pi + log_emissions[0]
    log_delta_0 = log_alpha_0

    # Run algorithms
    (log_alpha_z, log_delta_z), (log_alphas, log_deltas) = lax.scan(
        step,
        (log_alpha_0, log_delta_0),
        log_emissions[1:]
    )

    # Stack complete sequences
    #log_alphas = jnp.concatenate([log_alpha_0[None], log_alphas], axis=0)
    log_deltas = jnp.concatenate([log_delta_0[None], log_deltas], axis=0)

    return log_deltas, log_alpha_z

    # Get soft state assignments
    soft_paths = nn.softmax(log_deltas / temperature)

    loss = -jnp.sum(sp.special.logsumexp(log_alpha_z))

    return soft_paths, loss

def viterbi_softpath(log_deltas: Float[Array, "T S"], temperature: float = 1.0) -> Float[Array, "T"]:
    return nn.softmax(log_deltas / temperature)

def viterbi_loss(log_alpha_z: Float[Array, "S"]) -> Float[Array, "()"]:
    return -sp.special.logsumexp(log_alpha_z)



class HMM(nnx.Module):
    def __init__(self, num_states: int, num_features: int, rngs: nnx.Rngs):
        self.num_states = num_states
        self.num_features = num_features
        self.training = False

        # Emission parameters
        self.mean = nnx.Param(random.normal(rngs.params(), (num_states, num_features)))
        self.chol = nnx.Param(random.normal(rngs.params(), (num_states, num_features, num_features)))

        # Transition parameters  
        self.unnorm_transitions = nnx.Param(random.normal(rngs.params(), (num_states, num_states)))
        self.unnorm_initial_probs = nnx.Param(random.normal(rngs.params(), (num_states,)))

    @property
    def covs(self):
        C = self.chol.value
        return jnp.einsum("...ij,...kj->...ik", C, C)

    @property
    def log_A(self):
        transitions = self.unnorm_transitions.value
        log_norm = sp.special.logsumexp(transitions, axis=1, keepdims=True)
        return transitions - log_norm

    @property
    def log_pi(self):
        unnorm_log_pi = self.unnorm_initial_probs.value
        log_norm = sp.special.logsumexp(unnorm_log_pi)
        return unnorm_log_pi - log_norm

    def log_emmissions(self, obs):
        log_prob_fn = nnx.vmap(log_emission_prob, in_axes=(None, 0, 0))
        log_prob_fn = nnx.vmap(log_prob_fn, in_axes=(0, None, None))
        return log_prob_fn(obs, self.mean.value, self.covs)


    def __call__(self, obs, temperature: float = 1):
        log_emissions = self.log_emmissions(obs)

        log_deltas, log_alpha_z = viterbi(self.log_A, self.log_pi, log_emissions, temperature)

        if self.training:
            self.loss = Loss(viterbi_loss(log_alpha_z))

        return viterbi_softpath(log_deltas, temperature)

    def norm(self, obs):
        states = self(obs)

        # TODO: Normaliza las obervaciones basandose en los estados retornados
        # para ello utiliza las medias y covarianzas de los estados
        # y retorna las observaciones normalizadas
        # utiliza los estados para ponderar las normalizaciones

        return states
        

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
    
    model = HMM(num_states, num_features, rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))


    #%%
    # Train
    train_hmm(model, optimizer, sequences, true_states, num_epochs=2000)
    nnx.display(model)

    #%%
