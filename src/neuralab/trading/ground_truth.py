#%%
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Self

from flax import struct
from jax import grad, numpy as jnp
from jax import tree
from jaxtyping import Array, Float

from neuralab.trading.dataset import Dataset
from neuralab.trading.trends import Trends


@struct.dataclass
class GroundTruth(struct.PyTreeNode):

    @dataclass
    class Settings(struct.PyTreeNode):
        low_duration: int = 10
        high_duration: int = 30

        high_returns: float = 0.01
        low_returns: float = 0.005

        duration_score = jnp.array([1.0, 0.75, 0.5])
        returns_score = jnp.array([1.0, 0.75, 0.0])

        def trend_scores(self, trends: Trends):
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

        def side_trend_indices(self, trends: Trends):
            return (trends.returns < self.low_returns) | (
                (trends.duration > self.high_duration)
                & (trends.returns < self.high_returns)
            )

        def categorical_trend_indices(self, trends: Trends):
            side_trend_idx = self.side_trend_indices(trends)
            up_trend_idx = (~side_trend_idx) & (trends.direction > 0)
            down_trend_idx = (~side_trend_idx) & (trends.direction < 0)
            return up_trend_idx, side_trend_idx, down_trend_idx

    @classmethod
    def from_dataset(cls, dataset: Dataset, settings: GroundTruth.Settings = Settings()):
        all_trends = dataset.trends
        all_trend_scores = settings.trend_scores(all_trends)

        up_trend_idx, side_trend_idx, down_trend_idx = (
            settings.categorical_trend_indices(all_trends)
        )

        up_trends = all_trends[up_trend_idx]
        side_trends = all_trends[side_trend_idx]
        down_trends = all_trends[down_trend_idx]

        up_trend_scores = all_trend_scores[up_trend_idx]
        side_trend_cores = all_trend_scores[side_trend_idx]
        down_trend_scores = all_trend_scores[down_trend_idx]

        return cls(
            settings,
            dataset,
            all_trends,
            all_trend_scores,
            side_trends,
            up_trends,
            down_trends,
            side_trend_cores,
            up_trend_scores,
            down_trend_scores,
        )

    settings: Settings = struct.field(pytree_node=False)
    dataset: Dataset

    all_trends: Trends
    all_trend_scores: Float[Array, "all_trends"]

    up_trends: Trends
    side_trends: Trends    
    down_trends: Trends

    up_trend_scores: Float[Array, "up_trends"]
    side_trend_scores: Float[Array, "side_trends"]
    down_trend_scores: Float[Array, "down_trends"]


    @struct.dataclass
    class Labels(struct.PyTreeNode):
        one_hot: jnp.ndarray
        mask: jnp.ndarray

        def __len__(self):
            return len(self.one_hot)

        def __getitem__(self, *args) -> Self:
            return tree.map(lambda v: v.__getitem__(*args), self)

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]


    @cached_property
    def labels(self) -> Labels:
        mask = jnp.zeros(self.dataset.shape)

        def categorical_label(trends: Trends, trend_scores: jnp.ndarray):
            nonlocal mask

            mask = mask.at[trends.start_at, trends.batch, trends.market].add(
                trend_scores
            )
            mask = mask.at[trends.stop_at, trends.batch, trends.market].add(
                -trend_scores
            )

            label = jnp.zeros(self.dataset.shape)
            label = label.at[trends.start_at, trends.batch, trends.market].add(1)
            label = label.at[trends.stop_at, trends.batch, trends.market].add(-1)
            return jnp.cumsum(label, axis=0)

        one_hot = jnp.stack(
            [
                categorical_label(self.up_trends, self.up_trend_scores),
                categorical_label(self.side_trends, self.side_trend_scores),
                categorical_label(self.down_trends, self.down_trend_scores),
            ],
            axis=-1,
        )

        mask = jnp.cumsum(mask, axis=0)

        # calcula el balanceo de labels y compensa la mascara

        return self.Labels(one_hot, mask)


if __name__ == '__main__':
    dataset = Dataset.Ref.of("2024 1m").fetch()
    ground_truth = GroundTruth.from_dataset(dataset)
