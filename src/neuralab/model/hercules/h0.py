# %%
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
                "log_price",
                "log_volume",
                "log_bid_volume",
                "log_ask_volume",
            ]
        )

        @property
        def num_in_features(self):
            return len(self.in_features)

        @property
        def num_feed_features(self):
            sum_range = lambda L: (L**2 - L) // 2  # sum(range(L))
            return (
                self.num_in_features * self.num_ema_layers  # normalized features
                + self.num_in_features * self.num_ema_layers  # batch normalized vars
                + sum_range(self.num_ema_layers)  # price confussion matrix
                + sum_range(self.num_ema_layers)  # volume confussion matrix
                + sum_range(2 * self.num_ema_layers)  # imbalance confussion matrix
            )

        num_ema_layers: int = 4

        # num_hmm_layers: int = 4
        # num_hmm_features: int = 3
        # num_hmm_states: int = 8

        num_features = 64

    def __init__(
        self,
        settings=Settings(),
        *,
        rngs=nnx.Rngs(0),
    ):
        self.settings = settings

        self.ema = nl.EMStats(settings.num_ema_layers)

        self.var_norm = nnx.BatchNorm(
            settings.num_in_features * settings.num_ema_layers, rngs=rngs
        )

        self.in_proj = nnx.Linear(
            in_features=settings.num_feed_features,
            out_features=settings.num_features,
            rngs=rngs,
        )

        self.layers = nnx.Sequential(
            nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
            nl.Mamba.Settings(settings.num_features).build(rngs),
            nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
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

        self.logits_proj = nnx.Linear(settings.num_features, 3, rngs=rngs)

    def __call__(self, dataset: Dataset):
        x = dataset.features(
            *self.settings.in_features, axis=-1
        )  # [time, ..., feature]

        # EM Stats: avg var norm
        avg, var, nrm = self.ema.norm(x)  # [time, market, ..., feature, ema_layer]

        # confussion matrixes:
        # para cada una de las capas de ema:
        #   crea una matriz diferencial para cada una caracateristicas de entrada
        #  las ultimas dos capas (ask_vol y bid_vol) son concatenadas antes de crear la matriz
        #  por lo que tendremos 2 matrices de LxL y una de 2Lx2L
        #  devolviendo la parte triangular d ela matriz con jnp.triu
        # F = 2 * sum(range(L)) + sum(range(2*L))

        def diffussion_matrix(x):
            a = x[..., None, :]
            b = x[..., :, None]
            i, j = jnp.triu_indices(x.shape[-1], 1)
            return (a - b)[..., i, j]

        diffussion = jnp.concatenate(
            [
                rearrange(nrm, "t b m f l -> t b m (f l)"),
                self.var_norm(rearrange(var, "t b m f l -> t b m (f l)")),
                diffussion_matrix(avg[..., 0, :]),
                diffussion_matrix(avg[..., 1, :]),
                diffussion_matrix(
                    jnp.concatenate([avg[..., 2, :], avg[..., 3, :]], axis=-1)
                ),
            ],
            axis=-1,
        )

        feat = self.in_proj(diffussion)

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
        feat = self.layers(feat)

        # Sim
        logits = self.logits_proj(feat)

        return logits, diffussion, feat

    def loss(self, x: Dataset, y: Labels):
        logits, *_ = self(x)

        # MSE
        # loss = jnp.mean((logits - y.cats) ** 2, axis=-1) * y.mask
        
        # CrossEntropy
        loss = losses.softmax_cross_entropy(logits, y.cats) * y.mask

        return jnp.mean(loss)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # jax.config.update("jax_debug_nans", True)
    model = H0()
    trainer = Trainer(model)
    dataset = Dataset.load("tiny-fit")
    nnx.display(model)
    # %%
    loss, metrics, x, y = trainer(
        dataset,
        epochs=1000,
        #batch_slice=slice(0, 8),
    )
    nnx.display(model)

    # %%

    logits, diffusion, feat = model(dataset[:, 2:3])

    plt.pcolormesh(rearrange(diffusion[..., 0:1, :], "t b m f -> t (b m f)"))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(feat, "t b m f -> t (b m f)"))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(logits, "t b m a -> t (b m a)"))
    plt.colorbar()
    plt.show()

    # %%
    labels = trainer.get_labels()
    plt.pcolormesh(labels.cats[:, :, 0] * labels.mask[:, :, 0, None])
