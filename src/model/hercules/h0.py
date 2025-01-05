# %%
from math import prod
import optax
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp

from neuralab.ds.dataset import Dataset
from neuralab import nl


class H0(nnx.Module):

    @struct.dataclass
    class Params:
        num_ema_layers: int = 4
        ema_features: list[str] = struct.field(
            default_factory=lambda: [
                "log_volume",
                "log_imbalance",
                "diff_log_price",
            ]
        )
        num_hmm_layers: int = 4
        num_hmm_features: int = 3
        num_hmm_states: int = 8

        @property
        def num_ema_features(self):
            return len(self.ema_features)
        
        @property
        def num_features(self):
            return prod((self.num_hmm_layers, self.num_hmm_states))

    def __init__(
        self,
        params=Params(),
        *,
        rngs: nnx.Rngs,
    ):
        self.params = params

        self.ema = nl.EMA(params.num_ema_layers)

        self.hmm = nl.HMM.stack(
            params.num_hmm_layers,
            params.num_hmm_states,
            params.num_hmm_features,
            rngs=rngs,
        )

        self.mamba = nl.Mamba(
            nl.Mamba.Params(
                num_features=params.num_features,
            ),
            rngs=rngs,
        )

        self.mamba_norm = nnx.LayerNorm(params.num_features, rngs=rngs)

        self.sim_proj = nnx.Linear(params.num_features, 1, rngs=rngs)

    def __call__(self, ds: Dataset):
        x = ds.features(self.params.ema_features, axis=-1)  # [time, ..., feature]

        # EMA
        x = self.ema.std(x)  # [time, market, ..., feature, ema_layer]

        # HMM

        hmm_axes = nnx.StateAxes({nl.Loss: 1, ...: None})
        @nnx.vmap(in_axes=(hmm_axes, 1), out_axes=1)
        @nnx.vmap(in_axes=(0, -1), out_axes=-1)
        def hmm_vmap(hmm, features):
            return hmm(features)

        s = hmm_vmap(self.hmm, x)

        # flatten
        x = rearrange(s, "t ... s l -> t ... (s l)")

        # Mamba
        x = self.mamba_norm(x + self.mamba(x))


        # Sim
        x = self.sim_proj(x)

        return s, nnx.sigmoid(x)


@nnx.jit
def train_step(model, optimizer, dataset):
    """Single training step for H0 model"""

    def loss_fn(model):
        # Forward pass
        _, weights = model(dataset)
        print(weights.shape)

        sim_metrics = nl.sim(dataset.returns[..., None], weights, params=nl.SimParams())
        sim_loss = nl.sim_loss(sim_metrics)

        return sim_loss + nl.Loss.collect(model), sim_metrics

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
            print(f"Epoch {epoch}, Loss: {loss:.4f} {metrics}",)

        if epoch % 100 == 99:
            evaluate_h0(model, datasets[0][sli])
            model.train()


def evaluate_h0(model, dataset):
    """Evaluate H0 model and visualize results"""
    model.eval(training=False)
    soft_paths, _ = model(dataset)

    # Plot HMM state probabilities
    plt.figure(figsize=(12, 4))
    plt.title("HMM State Probabilities")
    # plt.imshow(rearrange(soft_paths[start:end], "t s l -> (l s) t"), aspect='auto', cmap='viridis')
    plt.pcolormesh(rearrange(soft_paths, "t ... s l -> (... l s) t"), cmap="viridis")
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.colorbar(label="State Probability")
    plt.tight_layout()
    plt.show()


# %%
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import optax
    from neuralab.ds.dataset import DATASETS

    rngs = nnx.Rngs(0)

    model = H0(rngs=rngs)  # Add these imports at the top
    optimizer = nnx.Optimizer(model, optax.adam(1e-3))

    # evaluate_h0(model, DATASETS[0])    # Add this function at the bottom

    # Train model
    # %%
    train_h0(model, optimizer, DATASETS, num_epochs=5000)

    # %%
    evaluate_h0(model, DATASETS[8][:1000])
    plt.plot(DATASETS[8][:1000].log_price)
    # %%
    nnx.display(model)
