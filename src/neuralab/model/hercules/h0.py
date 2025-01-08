# %%
from typing import Optional, Self
from optax import losses
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp

import pickle
from datetime import datetime

from neuralab import nl, settings
from neuralab.trading.dataset import Dataset, Feature
from neuralab.trading.trainer import Labels, Trainer


class Model(nnx.Module):
    @struct.dataclass
    class Settings: ...

    def save_state(self, name: Optional[str]):
        if name is None:
            name = datetime.now().isoformat()
        path = settings.storage_path(type(self), f"{name}-state.pkl")
        model_state = nnx.state(model)
        with path.open("wb") as f:
            pickle.dump(model_state, f)

    def load_state(self, name: str) -> Self:
        path = settings.storage_path(type(self), f"{name}-state.pkl")
        with path.open("rb") as f:
            model_state = pickle.load(f)
        graphdef, _ = nnx.split(self)
        return nnx.merge(graphdef, model_state)


class H0(Model):

    @struct.dataclass
    class Settings(Model.Settings):

        in_timeseries: list[Feature] = struct.field(
            default_factory=lambda: [
                "log_price",
                "log_volume",  # NOTE: dado que volume es la suma de bid y ask,
                # esta caracteristica podria descartarse en favor
                # de 'price', que es necesario para el computo
                # en algunos mercados
                "log_bid_volume",
                "log_ask_volume",
            ]
        )

        @property
        def num_in_timeseries(self):
            return len(self.in_timeseries)

        ema_max_t: int = 512

        @property
        def num_feed_features(self):
            sum_range = lambda L: (L**2 - L) // 2  # sum(range(L))
            return (
                self.num_in_timeseries * self.num_ema_layers  # normalized features
                + self.num_in_timeseries * self.num_ema_layers  # batch normalized vars
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

        self.ema = nl.EMStats(
            settings.num_ema_layers,
            max_t=settings.ema_max_t,
        )

        self.var_norm = nnx.BatchNorm(
            settings.num_in_timeseries * settings.num_ema_layers, rngs=rngs
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
        x = dataset.timeseries(
            *self.settings.in_timeseries, axis=-1
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

        def diffusion_matrix(x):
            a = x[..., None, :]
            b = x[..., :, None]
            i, j = jnp.triu_indices(x.shape[-1], 1)
            return (a - b)[..., i, j]

        diffusion = jnp.concatenate(
            [
                rearrange(nrm, "t b m f l -> t b m (f l)"),
                self.var_norm(rearrange(var, "t b m f l -> t b m (f l)")),
                diffusion_matrix(avg[..., 0, :]),
                diffusion_matrix(avg[..., 1, :]),
                diffusion_matrix(
                    jnp.concatenate([avg[..., 2, :], avg[..., 3, :]], axis=-1)
                ),
            ],
            axis=-1,
        )

        feat = self.in_proj(diffusion)

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

        # TODO: capa final colapsara la dimension de mercado unificando
        # todas las caracteristicas conjuntas, haciendo un feed forward
        # para luego componer los logits de cada cabeza de probabilidad

        # otra opcion es utilizar un multihead self attention con N cabezas.
        # embeddings en Q por mercado (Futures/Spot, binance kraken, etc...),

        logits = self.logits_proj(feat)

        return logits, diffusion, feat

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
    model = H0().load_state("test")
    trainer = Trainer(model)
    dataset = Dataset.load()

    # %%
    loss, metrics, x, y = trainer(
        dataset,
        epochs=1000,
        display_every=100,
        batch_size=2,
        # batch_slice=slice(0, 8),
    )

    # %%

    i = 5
    logits, diffusion, feat = model(x[:, i : i + 1])
    cats = y[:, i : i + 1].cats
    mask = y[:, i : i + 1].mask

    plt.pcolormesh(rearrange(diffusion[..., 0:1, :], "t b m f -> t (b m f)"))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(feat, "t b m f -> t (b m f)"))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(nnx.softmax(logits), "t b m a -> t (b m a)"))
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(cats * mask[..., None], "t b m a -> t (b m a)"))
    plt.colorbar()
    plt.show()
    # %%

    probs = nnx.softmax(logits)  # log_softmax??

    probs_long = probs[:, :, :, 0]
    probs_out = probs[:, :, :, 1]
    probs_short = probs[:, :, :, 2]
    # probs_hold = 1 - (probs_long + probs_out + probs_short)

    turnover = jnp.abs(jnp.diff(probs_long, axis=0)) + jnp(jnp.diff(probs_short, axis=0))
    returns = probs_long * dataset.returns - probs_short * dataset.returns

    plt.plot(jnp.cumsum(returns))

    # %%
    plt.pcolormesh(y.cats[:, :, 0] * y.mask[:, :, 0, None])

    # %% save model
    import pickle
    from datetime import datetime

    model_state = nnx.state(model)
    with open(f"{datetime.now()}.H0.pkl", "wb") as f:
        pickle.dump(model_state, f)

    # %%
    def plot_behaviour(
        model: H0, x: Dataset, y: Labels, batch_step: int = 5, market: int = 0
    ):
        import matplotlib.pyplot as plt

        i = batch_step
        m = market
        logits, diffusion, feat = model(x[:, i : i + 1])
        cats = y[:, i : i + 1].cats
        mask = y[:, i : i + 1].mask

        labels = []

        def append_label(label: str, count: int):
            labels.append(label)
            # labels.extend([' ' for i in range(count+1)])

        append_label("avg log price", model.settings.num_ema_layers)
        append_label("avg log volume", model.settings.num_ema_layers)
        append_label("avg log bid-volume", model.settings.num_ema_layers)
        append_label("avg log ask-volume", model.settings.num_ema_layers)

        append_label("norm log price", model.settings.num_ema_layers)
        append_label("norm log volume", model.settings.num_ema_layers)
        append_label("norm log bid-volume", model.settings.num_ema_layers)
        append_label("norm log ask-volume", model.settings.num_ema_layers)

        append_label("norm log price", model.settings.num_ema_layers)
        append_label("norm log volume", model.settings.num_ema_layers)
        append_label("norm log bid-volume", model.settings.num_ema_layers)
        append_label("norm log ask-volume", model.settings.num_ema_layers)

        fig, (avg_ax, norm_ax, price_diff_ax, volume_diff_ax, imbalance_ax) = (
            plt.subplots(ncols=1, nrows=5)
        )

        ax.set_yticklabels(labels)

        im = ax.pcolormesh(
            rearrange(diffusion[..., m : m + 1, :], "t b m f -> (b m f) t")
        )

        fig.colorbar(im, ax=ax)
        plt.show()

    plot_behaviour(model, x, y, batch_step=5, market=0)
