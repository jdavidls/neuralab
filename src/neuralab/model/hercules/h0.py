# %%
from typing import Optional, Self
from optax import losses
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp
import numpy as np

import pickle
from datetime import datetime

from neuralab import nl, settings
from neuralab.trading.dataset import Dataset, Feature
from neuralab.trading.sim import simulation
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
        out_shape = (2, 4,) # [ce/sim, feat]

        @property
        def num_out_logits(self) -> int:
            return int(np.prod(self.out_shape))

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

        self.feed_norm = nnx.BatchNorm(settings.num_feed_features, rngs=rngs)

        self.feed_proj = nnx.Linear(
            in_features=settings.num_feed_features,
            out_features=settings.num_features,
            rngs=rngs,
        )

        self.feat = nl.FeedForward(
            settings.num_feed_features,
            settings.num_features * 2,
            settings.num_features,
            residual=False,
            normalization=False,
            rngs=rngs,
        )

        self.layers = nnx.Sequential(
            # nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
            nl.Mamba.Settings(settings.num_features).build(rngs),
            # nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
        )

        # self.hmm = nl.HMM.stack(
        #     settings.num_hmm_layers,
        #     settings.num_hmm_states,
        #     settings.num_hmm_features,
        #     rngs=rngs,
        # )

        self.logits = nl.FeedForward(
            settings.num_features,
            settings.num_features * 2,
            settings.num_out_logits,
            residual=False,
            normalization=False,
            rngs=rngs,
        )

        self.logits_proj = nnx.Linear(settings.num_features, 3, rngs=rngs)

    def __call__(self, dataset: Dataset):
        x = dataset.timeseries(
            *self.settings.in_timeseries, axis=-1
        )  # [time, ..., feature]

        # Feed
        avg, var, nrm = self.ema.norm(x) # EM avg var norm

        def diff_matrix(x): # fn.diff_matrix and fn.diff_matrix_output(in) -> out { (in * in - in) / 2 }
            a = x[..., None, :]
            b = x[..., :, None]
            i, j = jnp.triu_indices(x.shape[-1], 1)
            return (a - b)[..., i, j]

        feed = jnp.concatenate(
            [
                rearrange(nrm, "t b m f l -> t b m (f l)"),
                rearrange(var, "t b m f l -> t b m (f l)"),
                diff_matrix(avg[..., 0, :]),
                diff_matrix(avg[..., 1, :]),
                diff_matrix(jnp.concatenate([avg[..., 2, :], avg[..., 3, :]], axis=-1)),
            ],
            axis=-1,
        )

        feed = self.feed_norm(feed)

        # Feat
        feat = self.feed_proj(feed)
        feat = self.layers(feat)

        # Logits
        logits = self.logits_proj(feat)

        return logits, feed, feat

    def loss(self, x: Dataset, y: Labels):
        logits, *_ = self(x)

        # MSE
        # loss = jnp.mean((logits - y.cats) ** 2, axis=-1) * y.mask

        # CrossEntropy
        # loss = losses.softmax_cross_entropy(logits, y.cats) * y.mask
        # return jnp.mean(loss)

        return simulation(x, logits).loss()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # jax.config.update("jax_debug_nans", True)
    model = H0().load_state("training")
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

    turnover = jnp.abs(jnp.diff(probs_long, axis=0)) + jnp.abs(
        jnp.diff(probs_short, axis=0)
    )
    returns = probs_long * dataset.returns - probs_short * dataset.returns

    plt.plot(jnp.cumsum(returns))
    plt.show()
    plt.plot(jnp.cumsum(turnover))
    plt.show()

    # %%
    plt.pcolormesh(y.cats[:, :, 0] * y.mask[:, :, 0, None])

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
