"""
This module contains the Trainer class, which is responsible for training the model.

"""

# %%
from math import isnan
from typing import Optional
from einops import rearrange
import optax
from jax import numpy as jnp, random, tree
from flax import nnx, struct

from neuralab.trading.dataset import Dataset, Trends
from tqdm.notebook import trange, tqdm

from neuralab import nl


@struct.dataclass
class Labels(struct.PyTreeNode):
    cats: jnp.ndarray
    mask: jnp.ndarray

    def __len__(self):
        return len(self.cats)

    def __getitem__(self, *args) -> Dataset:
        return tree.map(lambda v: v.__getitem__(*args), self)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Trainer(nnx.Module):
    @struct.dataclass
    class Settings(struct.PyTreeNode):
        low_duration: int = 10
        high_duration: int = 30

        high_returns: float = 0.01
        low_returns: float = 0.005

        duration_score = jnp.array([1.0, 0.75, 0.5])
        returns_score = jnp.array([1.0, 0.75, 0.0])

        def trend_initial_scores(self, trends: Trends):
            duration_score = jnp.select(
                [
                    trends.duration > self.high_duration,
                    trends.duration < self.low_duration,
                ],
                [
                    self.duration_score[0],
                    self.duration_score[2],
                ],
                self.duration_score[1],
            )

            returns_score = jnp.select(
                [
                    trends.returns > self.high_returns,
                    trends.returns < self.low_returns,
                ],
                [
                    self.returns_score[0],
                    self.returns_score[2],
                ],
                self.returns_score[1],
            )

            return duration_score * returns_score

        def is_side_trend(self, trends: Trends):
            return (trends.returns < self.low_returns) | (
                (trends.duration > self.high_duration)
                & (trends.returns < self.high_returns)
            )

        def categorical_trend_indices(self, trends: Trends):
            side_trend_idx = self.is_side_trend(trends)
            up_trend_idx = (~side_trend_idx) & (trends.direction > 0)
            down_trend_idx = (~side_trend_idx) & (trends.direction < 0)
            return up_trend_idx, side_trend_idx, down_trend_idx

    def __init__(self, model, settings: Settings = Settings(), rngs=nnx.Rngs(0)):
        self.settings = settings
        self.rngs = rngs
        self.model = model
        self.optimizer = nnx.Optimizer(model, optax.nadam(1e-3))

    # @nnx.jit
    def get_labels(self, dataset):

        all_trends = dataset.trends

        up_trend_idx, side_trend_idx, down_trend_idx = (
            self.settings.categorical_trend_indices(dataset.trends)
        )

        trend_scores = self.settings.trend_initial_scores(dataset.trends)

        up_trends = dataset.trends[up_trend_idx]
        side_trends = dataset.trends[side_trend_idx]
        down_trends = dataset.trends[down_trend_idx]
        up_trend_scores = trend_scores[up_trend_idx]
        side_trend_cores = trend_scores[side_trend_idx]
        down_trend_scores = trend_scores[down_trend_idx]

        mask = jnp.zeros(dataset.shape)

        def categorical_label(trends: Trends, trend_scores: jnp.ndarray):
            nonlocal mask
            mask = mask.at[trends.start_at, trends.batch, trends.market].add(
                trend_scores
            )
            mask = mask.at[trends.stop_at, trends.batch, trends.market].add(
                -trend_scores
            )

            label = jnp.zeros(dataset.shape)
            label = label.at[trends.start_at, trends.batch, trends.market].add(1)
            label = label.at[trends.stop_at, trends.batch, trends.market].add(-1)
            return jnp.cumsum(label, axis=0)

        labels = jnp.stack(
            [
                categorical_label(up_trends, up_trend_scores),
                categorical_label(side_trends, side_trend_cores),
                categorical_label(down_trends, down_trend_scores),
            ],
            axis=-1,
        )

        return Labels(labels, jnp.cumsum(mask, axis=0))

    def __call__(
        self,
        fit_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 1,
        batch_size: int = 2,
        batch_slice: Optional[slice] = None,
        batch_scan: bool = False,
    ):
        loss, metrics = None, None
        x = fit_dataset
        y = self.get_labels(fit_dataset)

        try:
            progbar = trange(epochs)

            for epoch in progbar:

                loss, metrics = self.epoch(x, y,
                    batch_size=batch_size,
                    batch_slice=batch_slice,
                    batch_scan=batch_scan,
                )

                progbar.set_description(f"Loss: {jnp.mean(loss):.3f}")
                # if eval_dataset is not None:
                # self.eval_step(eval_dataset)

        except KeyboardInterrupt:
            pass

        return loss, metrics, x, y

    def epoch(
        self,
        x,
        y,
        *,
        batch_size=None,
        batch_slice=None,
        batch_scan=False,
        batch_permute=True,
        batch_axis=1,
    ):

        # Batch slice
        if batch_slice is not None:
            x = x[:, batch_slice]
            y = y[:, batch_slice]

        # batch permutation
        if batch_permute:
            perm = random.permutation(self.rngs.train(), x.shape[1])
            x = tree.map(lambda v: v[:, perm], x)
            y = tree.map(lambda v: v[:, perm], y)

        # batch rearrange
        if batch_size is not None:
            assert x.shape[1] % batch_size == 0
            batch_count = x.shape[1] // batch_size

            def batch_rearrange(x):
                return rearrange(
                    x, "t (bl bc) ... -> bc t bl ...", bl=batch_size, bc=batch_count
                )

            x = tree.map(batch_rearrange, x)
            y = tree.map(batch_rearrange, y)
        else:
            batch_count = x.shape[1]

        # Batch scan
        if batch_scan:
            @nnx.scan(in_axes=0, out_axes=0)
            def batch_scanner(dataset: Dataset, labels: Labels):
                return self.train_step(dataset, labels)
            return batch_scanner(x, y)
        else:
            losses = []
            metrics = []
            progbar = tqdm(zip(x, y), total=batch_count)
            for n, (xx, yy) in enumerate(progbar):
                loss, mets = self.train_step(xx, yy)
                if jnp.isnan(loss):
                    raise ValueError("NaN loss")
                self.print_bacth_stats(progbar, loss, mets)
                losses.append(loss)
                metrics.append(mets)
            return jnp.array(losses), metrics # reduce metrics

    def print_bacth_stats(self, progbar, loss, metrics):
        progbar.set_description(f"Loss: {loss:.3f}")

    @nnx.jit
    def train_step(self, x, y):

        def loss_fn(model):
            loss = model.loss(x, y)

            safe_loss = jnp.nan_to_num(loss, nan=0, posinf=0, neginf=0)

            return safe_loss, (loss, nnx.pop(model, nl.Metric))

        # batch_scan
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (_, (loss, metrics)), grad = grad_fn(self.model)

        self.optimizer.update(grad)

        return loss, metrics


def categorical_crossentropy(logits, labels, mask, eps=1e-6):
    return -jnp.mean(jnp.sum(labels * jnp.log(logits + eps), axis=-1) * mask)


if __name__ == "__main__":
    from neuralab.model.hercules.h0 import H0

    ds = Dataset.load("default-fit")
    # ds = ds[:2**14].split_concat(8)

    rngs = nnx.Rngs(0)
    model = H0(rngs=rngs)
    trainer = Trainer(model, rngs=rngs)
    # %%
    loss, metrics = trainer(ds)

# %%
