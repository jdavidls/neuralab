# %%
import jax
import numpy as np
from chex import dataclass
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp

from neuralab import nl
from neuralab.trading.dataset import Dataset
from neuralab.trading.ground_truth import GroundTruth
from neuralab.trading.model.base_model import TradingModel
from neuralab.trading.trainer import Labels


class InputFeed(nnx.Module):

    def __init__(
        self,
        num_ema_layers: int,
        ema_max_t: int,
                normalization: bool = False,
        *,
        rngs=nnx.Rngs(0),
    ):
        self.ema = nl.EMStats(num_ema_layers, max_t=512)

        in_timeseries = 3

        self.num_out_features = (
            + num_ema_layers * 4  # normalized features
            + num_ema_layers * 4  # batch normalized vars
            + (num_ema_layers ** 2 - num_ema_layers) // 2  # price confussion matrix
            + (num_ema_layers ** 2 - num_ema_layers) // 2  # volume confussion matrix
            + (num_ema_layers ** 2 - num_ema_layers)  # imbalance confussion matrix
        )


        if normalization:
            self.norm = nnx.BatchNorm(, rngs=rngs)

        self.feed_proj = nnx.Linear(
            in_features=settings.num_feed_features,
            out_features=settings.num_features,
            rngs=rngs,
        )

    def __call__(self, dataset: Dataset):
        x = dataset.timeseries('log_price', "log_bid_volume",
                "log_ask_volume")  # [t a m f]

        avg, var, nrm = self.ema.norm(x)  # [t a m f e]

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

        if hasattr(self, "norm"):
            feed = self.norm(feed)

        return feed


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # jax.config.update("jax_debug_nans", True)
    model = H0()  # .load_state("training")
    trainer = nl.fit(model)

    dataset = Dataset.Ref.of("2021 1m").fetch()
    #with jax.checking_leaks():
    trainer.start(dataset, num_epochs=10000)


    # %%
    from neuralab.trading.ground_truth import GroundTruth

    ground_truth = GroundTruth.from_dataset(dataset, model.settings.training.ground_truth)
    sli = slice(0,1000)
    labels = ground_truth.labels[sli]   
    logits = model(dataset[sli])

    # plt.pcolormesh(rearrange(diffusion[..., 0:1, :], "t b m f -> t (b m f)"))
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(rearrange(feat, "t b m f -> t (b m f)"))
    # plt.colorbar()
    # plt.show()


    plt.pcolormesh(rearrange(labels.one_hot*labels.mask[..., None], "t a m p -> t (a m p)"))
    plt.colorbar()
    plt.show()


    plt.pcolormesh(rearrange(nnx.softmax(logits), "t a m p -> t (a m p)"))
    plt.colorbar()
    plt.show()

    # plt.pcolormesh(rearrange(cats * mask[..., None], "t b m a -> t (b m a)"))
    # plt.colorbar()
    # plt.show()
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
