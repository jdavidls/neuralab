# %%

from typing import Optional

from flax import nnx
from jax import lax, nn
from jax import numpy as jnp
from jax import random
from jax import scipy as sp
from jax.scipy import linalg
from jaxtyping import Array, Float

from neuralab.nl.common import Loss

LOG_TAU = jnp.log(2 * jnp.pi)


class GaussianStates(nnx.Module):
    def __init__(self, num_states: int, num_features: int, rngs: nnx.Rngs):
        self.num_states = num_states
        self.num_features = num_features

        self.mean = nnx.Param(random.normal(rngs.params(), (num_states, num_features)))
        self.chol = nnx.Param(
            random.normal(rngs.params(), (num_states, num_features, num_features))
        )
        # self.training = False

    @property
    def covs(self):
        C = self.chol.value
        return jnp.einsum("...ij,...kj->...ik", C, C)

    def log_emmission_prob(self, obs):
        log_prob_fn = nnx.vmap(log_emission_prob, in_axes=(None, 0, 0))
        log_prob_fn = nnx.vmap(log_prob_fn, in_axes=(0, None, None))
        return log_prob_fn(obs, self.mean.value, self.covs)

    def __call__(self, obs):
        return self.log_emmission_prob(obs)


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


class GaussianHMM(GaussianStates):
    def __init__(self, num_states: int, num_features: int, rngs: nnx.Rngs, temperature: float = 1.0):
        super().__init__(num_states, num_features, rngs)
        self._log_pi = nnx.Param(jnp.log(jnp.ones((num_states,)) / num_states))
        self._log_A = nnx.Param(jnp.log(jnp.ones((num_states, num_states)) / num_states))
        self.temperature = temperature
        self.training = False

    @property
    def log_A(self):
        """Returns normalized transition matrix in log space."""
        # Normalize each row using logsumexp
        transitions = self._log_A.value
        log_norm = sp.special.logsumexp(transitions, axis=1, keepdims=True)
        return transitions - log_norm

    @property
    def log_pi(self):
        """Returns normalized initial probabilities in log space."""
        unnorm_log_pi = self._log_pi.value
        log_norm = sp.special.logsumexp(unnorm_log_pi)
        return unnorm_log_pi - log_norm


    def forward_and_viterbi(self, obs):
        log_emissions = self.log_emmission_prob(obs)

        log_A = self.log_A  
        log_pi = self.log_pi

        def forward_and_viterbi_step(carry_n_1, log_emission_n):
            log_alpha_n_1, log_delta_n_1 = carry_n_1

            # Forward pass: α_t = P(emission_t|s_t) * Σ_s α_{t-1}(s) * P(s_t|s_{t-1})
            log_alpha_n = sp.special.logsumexp(log_alpha_n_1 + log_A, axis=1) + log_emission_n
            
            # Soft Viterbi: δ_t = P(emission_t|s_t) * softmax(δ_{t-1} + log A)
            log_delta_n = sp.special.logsumexp(log_delta_n_1 + log_A, axis=1) + log_emission_n  # [S]

            
            return (log_alpha_n, log_delta_n), (log_alpha_n, log_delta_n)

        # Initialize with π and first emission
        log_alpha_0 = log_pi + log_emissions[0]  # [S]
        log_delta_0 = log_alpha_0  # [S] 
        #soft_path_0 = nn.softmax(log_alpha_0 / self.temperature)  # [S]

        # Run forward algorithm for t=1:T
        _, (log_alphas, log_deltas) = lax.scan(
            forward_and_viterbi_step,
            (log_alpha_0, log_delta_0),
            log_emissions[1:]
        )

        # Add initial state to output
        log_alphas = jnp.concatenate([log_alpha_0[None], log_alphas], axis=0)
        log_deltas = jnp.concatenate([log_delta_0[None], log_deltas], axis=0)
        
        # 
        soft_paths = nn.softmax(log_deltas / self.temperature)  # [S]

        return log_alphas, soft_paths  # [T,S], [T,S]

    def __call__(self, obs):
        log_alphas, soft_paths = self.forward_and_viterbi(obs)

        #if self.training:
            #self.loss = Loss(-jnp.sum(sp.special.logsumexp(log_alphas[-1])))

        return soft_paths

    def loss(self, obs):
        log_alphas, soft_paths = self.forward_and_viterbi(obs)
        return -jnp.sum(sp.special.logsumexp(log_alphas[-1]))



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
                log_alphas, _ = model.forward_and_viterbi(sequence)
                return -jnp.sum(sp.special.logsumexp(log_alphas[-1]))

            losses = batch_loss(sequences)  # [B]
            return jnp.mean(losses)
            
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss
            
    # Training loop
    def train_hmm(model, optimizer, sequences, true_states, num_epochs=10000):

        model.train(training=True)
       
        for epoch in range(num_epochs):
            loss = train_step(model, optimizer, sequences)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

            if epoch % 100 == 0:
                eval_hmm(model, sequences[0], true_states[0])
                model.train(training=True)



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
    key = random.PRNGKey(42)
    sequences, true_states = generate_synthetic_data(key)
    
    num_states = 3
    num_features = 2
    rngs = nnx.Rngs(0)
    
    model = GaussianHMM(num_states, num_features, rngs, temperature=0.1)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))


    #%%
    # Train
    train_hmm(model, optimizer, sequences, true_states, num_epochs=10000)

    #%%
    # Evaluate
