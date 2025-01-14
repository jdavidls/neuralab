#%%
from __future__ import annotations

from functools import cached_property

from flax import struct
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Float, Int

from neuralab import track
from neuralab.trading.dataset import Dataset

@struct.dataclass
class Trends(struct.PyTreeNode):

    @staticmethod
    def concat(trends: list[Trends]):
        return tree.map(lambda *v: jnp.concat(v, axis=0), *trends)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Trends:
        assert dataset.logits is not None
        hard_logits = dataset.hard_logits

        events = jnp.diff(hard_logits, append=hard_logits[-1:], axis=0)

        t_idx, a_idx, m_idx = jnp.nonzero(events)

        assets_and_markets = [
            (a, m)
            for a in range(hard_logits.shape[1])
            for m in range(hard_logits.shape[2])
        ]

        with track.task(description="Computing trends", total=len(assets_and_markets)) as task:

            def trend_fn(a, m):
                task.update(advance=1)
                start_idx = t_idx[(a_idx == a) & (m_idx == m)]
                start_idx, stop_idx = start_idx[:-1], start_idx[1:]
                return cls(
                    dataset=dataset,
                    asset=jnp.full_like(stop_idx, a),
                    market=jnp.full_like(stop_idx, m),
                    start_at=start_idx,
                    stop_at=stop_idx,
                )

            return cls.concat([trend_fn(a, m) for a, m in assets_and_markets])

    dataset: Dataset = struct.field(pytree_node=False)

    asset: Int[Array, "trends"]
    market: Int[Array, "trends"]
    start_at: Int[Array, "trends"]
    stop_at: Int[Array, "trends"]

    @property
    def shape(self):
        return self.start_at.shape

    @property
    def batch(self):
        """We are using batch dim as a synonym for asset dim"""
        return self.asset

    @cached_property
    def duration(self) -> Int[Array, "trends"]:
        return self.stop_at - self.start_at

    @cached_property
    def start_log_price(self) -> Float[Array, "trends"]:
        return self.dataset.log_price[self.start_at, self.asset, self.market]

    @cached_property
    def stop_log_price(self) -> Float[Array, "trends"]:
        return self.dataset.log_price[self.stop_at, self.asset, self.market]

    @cached_property
    def returns(self) -> Float[Array, "trends"]:
        return jnp.abs(self.stop_log_price - self.start_log_price)

    @cached_property
    def direction(self) -> Float[Array, "trends"]:
        return jnp.sign(self.stop_log_price - self.start_log_price)

    def __len__(self):
        return len(self.start_at)

    def __getitem__(self, *args) -> Trends:
        return tree.map(lambda v: v.__getitem__(*args), self)

if __name__ == "__main__":
    dataset = Dataset.Ref.of("2024 1m").fetch()
    trends = Trends.from_dataset(dataset)
    print(trends)