# %%
import optax
from einops import rearrange, repeat
from flax import nnx
from jax import lax
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Float

from dataproxy.dataset import Dataset
from neuralab import nl
from neuralab.nl.common import Loss, Metric


class H0(nnx.Module):
    def __init__(
        self,
        num_ema_layers=4,
        ema_features=["log_volume", "log_volume_imbalance", "diff_log_price"],

        num_hmm_layers=4,
        num_hmm_features=3,
        num_hmm_states=5,
        *,
        rngs: nnx.Rngs,
    ):
        self.training = False
        self.num_ema_layers = num_ema_layers
        self.ema_features = ema_features
        self.num_ema_features = len(ema_features)

        self.ema = nl.EMA(num_ema_layers)


        self.num_hmm_layers = num_hmm_layers
        self.num_hmm_features = num_hmm_features

        self.hmm_proj = nnx.Linear(
            self.num_ema_layers * self.num_ema_features,
            self.num_hmm_layers * self.num_hmm_features,
            rngs = rngs
        )

        self.hmm = nl.HMM.stack(num_hmm_layers, num_hmm_states, num_hmm_features, rngs=rngs)

        self.hmm_loss = Loss(None)

    def __call__(self, ds: Dataset):
        x = jnp.stack(
            [getattr(ds, feature) for feature in self.ema_features], axis=-1
        )

        x = self.ema.std(x)

        #x = rearrange(x, "t f l -> t (f l)", f=self.num_ema_features, l=self.num_ema_layers)
        #x = self.hmm_proj(x)
        #x = rearrange(x, "t (f l) -> t f l", f=self.num_hmm_features, l=self.num_hmm_layers)

        # HMM
        soft_paths = nnx.vmap(
            lambda hmm, x: hmm(x), in_axes=(0, -1), out_axes=-1
        )(self.hmm, x)

        return soft_paths
    


@nnx.jit
def train_step(model, optimizer, dataset):
    """Single training step for H0 model"""

    def loss_fn(model):
        # Forward pass
        model(dataset)

        return Loss.collect(model), {}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    optimizer.update(grads)
    return loss, metrics


def train_h0(model, optimizer, datasets, num_epochs=10000, sli=slice(0, 40320 // 16)):
    """Training loop for H0 model"""

    for epoch in range(num_epochs):
        # Cycle through available datasets
        dataset = datasets[epoch % len(datasets)]
        model.train(training=True)
        loss, metrics = train_step(model, optimizer, dataset[sli])

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

        if epoch % 100 == 99:
            evaluate_h0(model, datasets[0][sli])
            model.train()


def evaluate_h0(model, dataset):
    """Evaluate H0 model and visualize results"""
    model.eval(training=False)
    soft_paths = model(dataset)

    # Plot HMM state probabilities
    plt.figure(figsize=(12, 4))
    plt.title("HMM State Probabilities")
    # plt.imshow(rearrange(soft_paths[start:end], "t s l -> (l s) t"), aspect='auto', cmap='viridis')
    plt.pcolormesh(rearrange(soft_paths, "t s l -> (l s) t"), cmap="viridis")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.colorbar(label="State Probability")
    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import optax
    from dataproxy.dataset import DATASETS

    rngs = nnx.Rngs(0)

    model = H0(rngs=rngs)  # Add these imports at the top
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    # evaluate_h0(model, DATASETS[0])    # Add this function at the bottom

    # Train model
    #%%
    train_h0(model, optimizer, DATASETS, num_epochs=5000)

    # %%
    evaluate_h0(model, DATASETS[8][:1000])
    plt.plot(DATASETS[8][:1000].log_price)
    # %%
    nnx.display(model)
