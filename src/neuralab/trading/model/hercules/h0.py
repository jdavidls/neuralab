# %%
import pickle
import jax
import numpy as np
from chex import dataclass
from einops import rearrange
from flax import nnx, struct
from jax import numpy as jnp
import optax

from neuralab import fn, nl
from neuralab.nl.common import State
from neuralab.trading.dataset import Dataset

from neuralab.trading.ground_truth import GroundTruth
from neuralab.trading.model.trading_model import TradingModel
from neuralab.trading.trainer import Labels


class H0(TradingModel):

    @dataclass()
    class Settings(TradingModel.Settings):

        in_timeseries: list[Dataset.Feature] = struct.field(
            default_factory=lambda: [
                "log_price",
                # "log_volume",  # NOTE: dado que volume es la suma de bid y ask,
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
            return (
                +self.num_in_timeseries * self.num_ema_layers  # normalized features
                + self.num_in_timeseries * self.num_ema_layers  # batch normalized vars
                + fn.diff_matrix_num_outputs(self.num_ema_layers)  # price diff matrix
                + fn.diff_matrix_num_outputs(self.num_ema_layers)  # volume diff matrix
                + fn.diff_matrix_num_outputs(
                    2 * self.num_ema_layers
                )  # imbalance diff matrix
            )

        num_ema_layers: int = 4

        # num_hmm_layers: int = 4
        # num_hmm_features: int = 3
        # num_hmm_states: int = 8
        num_features = 64
        num_out_logits = 3

    settings: Settings

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

        self.feat_proj = nnx.Linear(
            settings.num_feed_features,
            settings.num_features,
            rngs=rngs,
        )

        # self.feat_proj = nl.TriactForward(
        #     settings.num_feed_features, settings.num_features, rngs=rngs
        # )

        self.layers = nnx.Sequential(
            # nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
            nl.Mamba.Settings(settings.num_features).build(rngs),
            # nl.FeedForward(settings.num_features, 2.0, rngs=rngs),
        )

        self.logits_proj = nnx.Linear(
            settings.num_features,
            len(settings.head_losses) * 3,
            rngs=rngs,
        )

        self.feed = nnx.Cache(None)
        self.feat = nnx.Cache(None)

    @nnx.jit()
    def __call__(self, dataset: Dataset):
        x = dataset.timeseries(
            *self.settings.in_timeseries, axis=-1
        )  # [time, ..., feature]

        # Feed
        avg, var, nrm = self.ema.norm(x)  # EM avg var norm

        feed = jnp.concatenate(
            [
                rearrange(nrm, "t b m f l -> t b m (f l)"),
                rearrange(var, "t b m f l -> t b m (f l)"),
                fn.diff_matrix(avg[..., 0, :]),
                fn.diff_matrix(avg[..., 1, :]),
                fn.diff_matrix(
                    jnp.concatenate([avg[..., 2, :], avg[..., 3, :]], axis=-1)
                ),
            ],
            axis=-1,
        )

        feed = self.feed_norm(feed)
        self.feed = nnx.Cache(feed)

        # Feat
        feat = self.feat_proj(feed)
        feat = self.layers(feat)
        self.feat = nnx.Cache(feat)

        # Logits
        logits = self.logits_proj(feat)

        return rearrange(logits, "t b m (h f) -> t b m h f", h=self.settings.num_heads)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # jax.config.update("jax_debug_nans", True)
    model = H0(
        settings=H0.Settings(
            trainer=H0.Trainer.Settings(
                optimizer=optax.adam(1e-2),
            )
        )
    )
    
    # %%

    with open("model.pkl", "rb") as f:
        state = pickle.load(f)
        del state["training"]
        del state["ema"]["state"]
        state["ema"]["ema_state"] = nnx.VariableState(type=State, value=None)
        state["ema"]["emv_state"] = nnx.VariableState(type=State, value=None)
        state["layers"]["layers"][0]["ssm"]["state"] = nnx.VariableState(
            type=State, value=None
        )

    graph, _ = nnx.split(model)
    model = nnx.merge(graph, state)

    # %%
    dataset = Dataset.Ref.of("2021 to 2023 1m").fetch()
    trainer = nl.fit(model)
    trainer(dataset, num_epochs=1000)

    # %% evaluation
    evaluation_set = Dataset.Ref.of("2023 1m").fetch()

    # %%

    # %%
    from neuralab.trading.evaluation import evaluate

    # @nnx.jit()
    def eval_perf(model, dataset):
        logits = model(dataset)
        return evaluate(logits[..., 0, :], dataset).total_performance

    perf = jnp.array(
        [eval_perf(model, batch) for batch in evaluation_set.batched(2048)]
    )

    # %%

    ev_set = evaluation_set[20000:22048]
    logits = model(ev_set)[..., 1, :]
    eval = evaluate(logits, ev_set)

    price = ev_set.log_price[:, 0, 0]

    plt.plot(
        (price - jnp.min(price)) / (jnp.max(price) - jnp.min(price)), color="green"
    )
    plt.pcolormesh(rearrange(nnx.softmax(logits[:, 0, 0]), "t p -> 1 t p"))
    plt.colorbar()
    plt.show()

    # %% Surgery

    # show feed
    plt.pcolormesh(rearrange(model.feed.value[:100], "t a m f -> t (a m f)"))
    plt.colorbar()
    plt.show()

    # %%
    from neuralab.trading.ground_truth import GroundTruth

    ground_truth = GroundTruth.from_dataset(
        dataset, model.settings.trainer.ground_truth
    )
    sli = slice(0, 1000)
    labels = ground_truth.labels[sli]
    logits = model(dataset[sli])

    # plt.pcolormesh(rearrange(diffusion[..., 0:1, :], "t b m f -> t (b m f)"))
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(rearrange(feat, "t b m f -> t (b m f)"))
    # plt.colorbar()
    # plt.show()

    plt.pcolormesh(
        rearrange(labels.one_hot * labels.mask[..., None], "t a m p -> t (a m p)")
    )
    plt.colorbar()
    plt.show()

    plt.pcolormesh(rearrange(nnx.softmax(logits), "t a m h p -> t (h a m p)"))
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
