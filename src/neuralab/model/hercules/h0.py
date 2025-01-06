# %%
from math import prod
from optax import losses
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp


from neuralab import nl
from neuralab.trading.dataset import Dataset, Feature
from neuralab.trading.trainer import Labels, Trainer


class H0(nnx.Module):

    @struct.dataclass
    class Settings:

        in_features: list[Feature] = struct.field(
            default_factory=lambda: [
                "log_volume",
                "log_imbalance",
                "diff_log_price",
            ]
        )

        @property
        def num_in_features(self):
            return len(self.in_features)

        num_ema_layers: int = 4

        # num_hmm_layers: int = 4
        # num_hmm_features: int = 3
        # num_hmm_states: int = 8

        num_features = 64

    def __init__(
        self,
        settings=Settings(),
        *,
        rngs = nnx.Rngs(0),
    ):
        self.settings = settings

        self.ema = nl.EMA(settings.num_ema_layers)

        self.in_proj = nnx.Linear(
            settings.num_in_features * settings.num_ema_layers,
            settings.num_features,
            rngs=rngs,
        )

        self.layers = nnx.Sequential(
            nl.Mamba.Settings(settings.num_features).build(rngs),
        )

        # self.hmm = nl.HMM.stack(
        #     settings.num_hmm_layers,
        #     settings.num_hmm_states,
        #     settings.num_hmm_features,
        #     rngs=rngs,
        # )

        # self.mamba = nl.Mamba(
        #     nl.Mamba.Settings(
        #         num_features=settings.num_features,
        #     ),
        #     rngs=rngs,
        # )

        # self.mamba_norm = nnx.LayerNorm(settings.num_features, rngs=rngs)

        self.sim_proj = nnx.Linear(settings.num_features, 3, rngs=rngs)

    def __call__(self, ds: Dataset):
        x = ds.features(self.settings.in_features, axis=-1)  # [time, ..., feature]

        # EMA
        x = self.ema.std(x)  # [time, market, ..., feature, ema_layer]

        x = self.in_proj(rearrange(x, "t ... f l -> t ... (f l)"))

        # # HMM

        # hmm_axes = nnx.StateAxes({nl.Loss: 1, ...: None})
        # @nnx.vmap(in_axes=(hmm_axes, 1), out_axes=1)
        # @nnx.vmap(in_axes=(0, -1), out_axes=-1)
        # def hmm_vmap(hmm, features):
        #     return hmm(features)

        # s = hmm_vmap(self.hmm, x)

        # # flatten
        # x = rearrange(s, "t ... s l -> t ... (s l)")

        # Mamba
        x = self.layers(x)

        # Sim
        x = self.sim_proj(x)

        return x
    
    def loss(self, x: Dataset, y: Labels):
        logits = self(x)

        sce = y.mask * losses.safe_softmax_cross_entropy(logits, y.cats)

        return -jnp.mean(sce)

if __name__ == "__main__":
    from neuralab.model.hercules.h0 import H0

    model = H0()
    trainer = Trainer(model)
    dataset = Dataset.load('default-fit')
    trainer(dataset)

    
# %%
