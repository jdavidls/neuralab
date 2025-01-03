# %%
import optax
from einops import rearrange
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float

from dataproxy.dataset import Dataset
from neuralab import nl

class H0(nnx.Module):
    def __init__(
        self,
        feed_layers = 4,
        feed_vocab = 5**2,
        *,
        rngs: nnx.Rngs,
    ):
        # Input FEED
        self.ema = nl.EMA(feed_layers)

        feed_features = ['std_volume', 'std_volume_imbalance']
        num_feed_features = len(feed_features)

        self.hmm = nl.HMM.stack(feed_layers, feed_vocab, num_feed_features, rngs=rngs)


    def __call__(self, ds: Dataset):
        v = self.ema.standarize(ds.log_volume)
        u = self.ema.standarize(ds.log_volume_imbalance, is_stationary=True)
        x = jnp.stack([v, u], axis=-1)


        # HMM
        soft_paths, hmm_losses = nnx.vmap(lambda hmm, x: hmm(x), in_axes=(0, -2), out_axes=-1)(self.hmm, x)

        return soft_paths, hmm_losses


@nnx.jit
def train_step(model, optimizer, dataset):
    """Single training step for H0 model"""
    def loss_fn(model):
        # Forward pass
        soft_paths, hmm_losses = model(dataset)
        # Total loss is the sum of HMM losses
        loss = jnp.mean(hmm_losses)
        return loss, {"hmm_loss": loss}
        
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return metrics

def train_h0(model, optimizer, datasets, num_epochs=10000):
    """Training loop for H0 model"""
    
    
    for epoch in range(num_epochs):
        # Cycle through available datasets
        dataset = datasets[epoch % len(datasets)]
        model.train(training=True)
        metrics = train_step(model, optimizer, dataset)
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {metrics['hmm_loss']:.4f}")
            
        if epoch % 100 == 99:
            evaluate_h0(model, dataset)
            model.train()

def evaluate_h0(model, dataset):
    """Evaluate H0 model and visualize results"""
    model.eval(training=False)
    soft_paths, _ = model(dataset)
    
    # Plot HMM state probabilities
    plt.figure(figsize=(12, 4))
    plt.title("HMM State Probabilities")
    plt.imshow(rearrange(soft_paths[:1000], "t s l -> (l s) t"), aspect='auto', cmap='viridis')
    #plt.pcolormesh(rearrange(soft_paths[:1000], "t s l -> (l s) t"), cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.colorbar(label='State Probability')
    plt.tight_layout()
    plt.show()

# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import optax
    from dataproxy.dataset import DATASETS

    rngs = nnx.Rngs(0)

    model = H0(rngs=rngs)    # Add these imports at the top
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    evaluate_h0(model, DATASETS[0])    # Add this function at the bottom
    #%%
    # Train model

    train_h0(model, optimizer, DATASETS, num_epochs=10)
