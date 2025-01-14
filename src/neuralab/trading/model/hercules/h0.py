# %%
import jax
import numpy as np
from chex import dataclass
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp

from neuralab import nl
from neuralab.trading.dataset import Dataset
from neuralab.trading.model.base_model import TradingModel
from neuralab.trading.trainer import Labels


class H0(TradingModel):


    @dataclass(frozen=True)
    class Settings(TradingModel.Settings):

        in_timeseries: list[Dataset.Feature] = struct.field(
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
        num_out_logits = 3

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
            2.0,
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

        self.logits = nl.FeedForward(
            settings.num_features,
            2.0,
            settings.num_out_logits,
            residual=False,
            normalization=False,
            rngs=rngs,
        )

    def __call__(self, dataset: Dataset):
        x = dataset.timeseries(
            *self.settings.in_timeseries, axis=-1
        )  # [time, ..., feature]

        # Feed
        avg, var, nrm = self.ema.norm(x)  # EM avg var norm

        def diff_matrix(
            x,
        ):  # fn.diff_matrix and fn.diff_matrix_output(in) -> out { (in * in - in) / 2 }
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
        logits = self.logits(feat)

        return logits


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # jax.config.update("jax_debug_nans", True)
    model = H0()  # .load_state("training")
    trainer = nl.fit(model)

    dataset = Dataset.Ref.of("2021 1m").fetch()
    #with jax.checking_leaks():
    trainer.start(dataset, num_epochs=10000)


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

# %%
