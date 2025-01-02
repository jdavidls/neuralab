#%%
import torch
import torch.nn.functional as F

TAU = torch.tensor(2 * torch.pi)

class HMM:
    def __init__(self, num_states, num_features):
        self.num_states = num_states
        self.num_features = num_features

        # Initialize model parameters
        self.pi = torch.nn.Parameter(torch.log(torch.ones(num_states) / num_states))  # Initial state probabilities (log-space)
        self.A = torch.nn.Parameter(torch.log(torch.ones(num_states, num_states) / num_states))  # Transition probabilities (log-space)
        self.mu = torch.nn.Parameter(torch.randn(num_states, num_features))  # Mean of Gaussian emissions
        self.sigma = torch.nn.Parameter(torch.ones(num_states, num_features))  # Std dev of Gaussian emissions

    def log_emission_prob(self, obs):
        """Compute log-probabilities of observations given states."""
        obs = obs.unsqueeze(1)  # Shape (T, 1, num_features)
        mu = self.mu.unsqueeze(0)  # Shape (1, num_states, num_features)
        sigma = self.sigma.unsqueeze(0)  # Shape (1, num_states, num_features)
        log_prob = -0.5 * torch.sum(((obs - mu) / sigma)**2 + 2 * torch.log(sigma) + torch.log(TAU), dim=2)
        return log_prob  # Shape (T, num_states)

    def forward_algorithm(self, obs):
        """Compute the forward probabilities using the Forward algorithm."""
        T = obs.shape[0]
        log_alpha = torch.zeros(T, self.num_states)

        # Initial step
        log_alpha[0] = self.pi + self.log_emission_prob(obs[0].unsqueeze(0))

        # Recursion
        for t in range(1, T):
            log_alpha[t] = torch.logsumexp(log_alpha[t - 1].unsqueeze(1) + self.A, dim=0) + self.log_emission_prob(obs[t].unsqueeze(0))

        return log_alpha

    def backward_algorithm(self, obs):
        """Compute the backward probabilities using the Backward algorithm."""
        T = obs.shape[0]
        log_beta = torch.zeros(T, self.num_states)

        # Initial step (at T)
        log_beta[T - 1] = 0

        # Recursion
        for t in range(T - 2, -1, -1):
            log_beta[t] = torch.logsumexp(self.A + self.log_emission_prob(obs[t + 1].unsqueeze(0)) + log_beta[t + 1].unsqueeze(0), dim=1)

        return log_beta

    def baum_welch(self, obs, max_iter=100, tol=1e-4):
        """Train the HMM using the Baum-Welch algorithm."""
        T = obs.shape[0]
        prev_log_likelihood = float('-inf')
        for iteration in range(max_iter):
            # E-step
            log_alpha = self.forward_algorithm(obs)
            log_beta = self.backward_algorithm(obs)

            log_gamma = log_alpha + log_beta
            log_gamma -= torch.logsumexp(log_gamma, dim=1, keepdim=True)  # Normalize

            log_xi = log_alpha[:-1].unsqueeze(2) + self.A.unsqueeze(0) + self.log_emission_prob(obs[1:]).unsqueeze(1) + log_beta[1:].unsqueeze(1)
            log_xi -= torch.logsumexp(log_xi.view(-1, self.num_states * self.num_states), dim=0).view(self.num_states, self.num_states)  # Normalize

            # M-step
            self.pi.data = log_gamma[0]
            self.A.data = torch.logsumexp(log_xi, dim=0) - torch.logsumexp(log_gamma[:-1], dim=0, keepdim=True)

            weights = torch.exp(log_gamma).unsqueeze(2)  # Shape (T, num_states, 1)
            self.mu.data = torch.sum(weights * obs.unsqueeze(1), dim=0) / torch.sum(weights, dim=0)
            self.sigma.data = torch.sqrt(torch.sum(weights * (obs.unsqueeze(1) - self.mu.data)**2, dim=0) / torch.sum(weights, dim=0))

            # Log-likelihood for convergence
            log_likelihood = torch.sum(torch.logsumexp(log_alpha[-1], dim=0))
            if iteration > 0 and torch.abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood

    def gradient_based_training(self, obs, max_iter=100, lr=1e-2):
        """Train the HMM using gradient-based optimization with autodifferentiation."""
        optimizer = torch.optim.Adam([self.pi, self.A, self.mu, self.sigma], lr=lr)

        for iteration in range(max_iter):
            optimizer.zero_grad()

            # Forward algorithm for log-likelihood computation
            log_alpha = self.forward_algorithm(obs)
            log_likelihood = torch.sum(torch.logsumexp(log_alpha[-1], dim=0))

            # Loss is the negative log-likelihood
            loss = -log_likelihood
            loss.backward()

            optimizer.step()

    def viterbi(self, obs):
        """Find the most likely state sequence using the Viterbi algorithm."""
        T = obs.shape[0]
        log_delta = torch.zeros(T, self.num_states)
        psi = torch.zeros(T, self.num_states, dtype=torch.long)
        # psi = torch.zeros(T, self.num_states)

        # Initial step
        log_delta[0] = self.pi + self.log_emission_prob(obs[0].unsqueeze(0))

        # Recursion
        for t in range(1, T):
            max_val, argmax_val = torch.max(log_delta[t - 1].unsqueeze(1) + self.A, dim=0)
            log_delta[t] = max_val + self.log_emission_prob(obs[t].unsqueeze(0))
            psi[t] = argmax_val

        # Debugging: Print intermediate values
        #print(f"log_delta: {log_delta}")
        print(f"psi: {psi}")

        # Backtracking
        states = torch.zeros(T, dtype=torch.long)
        states[-1] = torch.argmax(log_delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate some data (T observations with num_features)
    T = 100
    num_features = 2
    obs = torch.randn(T, num_features)


    # Create and train the HMM
    hmm1 = HMM(num_states=3, num_features=num_features)
    hmm1.gradient_based_training(obs)

    # Decode the state sequence
    states = hmm1.viterbi(obs)
    print("Decoded states:", states)


    # Create and train the HMM
    hmm2 = HMM(num_states=3, num_features=num_features)
    hmm2.baum_welch(obs)

    # Decode the state sequence
    states = hmm2.viterbi(obs)
    print("Decoded states:", states)


#%%


# Function to generate synthetic data
def generate_synthetic_data(num_states, num_features, num_samples):
    """Generates synthetic data for an HMM with multiple transition phases."""
    torch.manual_seed(42)

    # Define transition probabilities and initial probabilities
    transition_probs = torch.tensor([[0.7, 0.2, 0.1],
                                      [0.3, 0.4, 0.3],
                                      [0.2, 0.3, 0.5]])
    initial_probs = torch.tensor([0.5, 0.3, 0.2])

    # Emission parameters (mean and std deviation for each state)
    emission_means = torch.tensor([[2.0, 3.0], [0.0, 0.0], [-2.0, -3.0]])
    emission_stds = torch.tensor([[0.5, 0.5], [1.0, 1.0], [0.5, 0.5]])

    # Sample initial state
    states = []
    observations = []

    state = torch.multinomial(initial_probs, 1).item()
    states.append(state)

    # Generate observations
    for _ in range(num_samples):
        mean = emission_means[state]
        std = emission_stds[state]
        obs = torch.randn(num_features) * std + mean
        observations.append(obs)

        state = torch.multinomial(transition_probs[state], 1).item()
        states.append(state)

    return torch.tensor(states[:-1]), torch.stack(observations)

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)

    # Generate synthetic data
    num_states = 3
    num_features = 2
    num_samples = 100
    true_states, observations = generate_synthetic_data(num_states, num_features, num_samples)

    # Train HMM
    hmm = HMM(num_states=num_states, num_features=num_features)
    hmm.gradient_based_training(observations)

    # Decode states
    predicted_states = hmm.viterbi(observations)

    print("True states:", true_states)
    print("Predicted states:", predicted_states)

#%%
