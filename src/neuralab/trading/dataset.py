# %%
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import cached_property
from gc import collect
from pathlib import Path
from typing import IO, Annotated, Any, Callable, Literal, Optional, Self

from flax import struct
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Float
from pandas import DataFrame
from typer import Argument, Option, Typer

from neuralab.resource import Resource, fetch_tree
from neuralab.trading.dataframe import TradeSampleDataFrame
from neuralab.trading.knowledge import (
    DEFAULT_MARKETSET,
    DEFAULT_SYMBOLSET,
    MARKETSET_NAMES,
    SYMBOLSET_NAMES,
    MarketSet,
    SymbolSet,
)
from neuralab.utils.base_dataset import BaseDataset, BatchIterable
from neuralab.utils.timeutils import TimeRange


@struct.dataclass
class Dataset(Resource, BaseDataset):

    @dataclass(frozen=True)
    class Ref[T: Dataset](Resource.Ref[T]):

        @classmethod
        def of(
            cls,
            time_range: TimeRange.Parseable,
            symbol_set: SymbolSet = DEFAULT_SYMBOLSET,
            market_set: MarketSet = DEFAULT_MARKETSET,
        ):
            return cls(TimeRange.of(time_range), symbol_set, market_set)

        time_range: TimeRange
        symbol_set: SymbolSet
        market_set: MarketSet

        @property
        def symbols(self):
            return sorted(self.symbol_set)

        @property
        def markets(self):
            return sorted(self.market_set)

        @property
        def symbolset_name(self):
            if self.symbol_set in SYMBOLSET_NAMES:
                return SYMBOLSET_NAMES[self.symbol_set]
            return "-".join(sorted(self.symbol_set))

        @property
        def marketset_name(self):
            if self.market_set in MARKETSET_NAMES:
                return MARKETSET_NAMES[self.market_set]
            return "+".join(sorted(self.market_set))

        @property
        def path(self):
            return (
                Path("default")  # reservado para logits settings
                / f"{self.marketset_name}"
                / f"{self.symbolset_name}"
                / f"{self.time_range}"
            )

        @property
        def type(self):
            return Dataset

        @property
        def path_suffix(self):
            return f".pkl"

        def load(self, buffer: IO[bytes]) -> T:
            return pickle.load(buffer)

        def save(self, buffer: IO[bytes], dataset: T):
            pickle.dump(dataset, buffer)

        def make(self):
            symbols = fetch_tree(
                [
                    [
                        TradeSampleDataFrame(
                            symbol=s, market=m, time_range=self.time_range
                        )
                        for m in self.markets
                    ]
                    for s in self.symbols
                ]
            )

            dataset = Dataset.concat(
                [
                    Dataset.concat(
                        [Dataset.from_dataframe(self, df) for df in markets], axis=2
                    )
                    for markets in symbols
                ],
                axis=1,
            )

            del symbols
            collect()

            return dataset.fit_logits()

    @classmethod
    def from_dataframe(cls, ref: Ref, df: DataFrame):

        def df_timeserie(nm: str):
            return df[nm].to_numpy()[:, None, None]

        price = df_timeserie("vwap")
        vol = df_timeserie("vol")
        ask_vol = df_timeserie("ask_vol")
        bid_vol = df_timeserie("bid_vol")

        log_price = jnp.log(price)

        return cls(
            ref=ref,
            log_price=log_price,
            returns=jnp.diff(log_price, append=log_price[-1:], axis=0),
            diff_log_price=jnp.diff(log_price, prepend=log_price[:1], axis=0),
            log_volume=jnp.log1p(vol),
            log_ask_volume=jnp.log1p(ask_vol),
            log_bid_volume=jnp.log1p(bid_vol),
        )

    type Feature = Literal[
        "log_price",
        "returns",
        "diff_log_price",
        "log_volume",
        "log_imbalance",
    ]

    ref: Ref[Dataset] = struct.field(pytree_node=False)  # type: ignore

    log_price: Float[Array, "time asset market"]
    returns: Float[Array, "time asset market"]
    diff_log_price: Float[Array, "time asset market"]
    log_volume: Float[Array, "time asset market"]
    log_ask_volume: Float[Array, "time asset market"]
    log_bid_volume: Float[Array, "time asset market"]

    logits: Optional[Float[Array, "time asset market"]] = None

    @property
    def log_imbalance(self) -> Float[Array, "time asset market"]:
        return self.log_ask_volume - self.log_bid_volume

    @property
    def hard_logits(self) -> Float[Array, "time asset market"]:
        assert self.logits is not None
        return jnp.sign(self.logits)

    @property
    def shape(self):
        return self.log_price.shape

    def __len__(self):
        return len(self.log_price)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, *args) -> Self:
        return tree.map(lambda v: v.__getitem__(*args), self)

    def timeseries(self, *feature_names: Feature, axis=-1):
        return jnp.stack(
            [getattr(self, feature) for feature in feature_names], axis=axis
        )

    def batch_iter(self, batch_size: int) -> BatchIterable[Dataset]:
        return BatchIterable(self, batch_size)


    @staticmethod
    def concat(datasets: list[Dataset], axis=1) -> Dataset:
        return tree.map(lambda *v: jnp.concat(v, axis=axis), *datasets)

    # def split(self, indices_or_sections, axis=0):
    # return tree.map(lambda v: jnp.split(v, indices_or_sections, axis=axis), self)

    def split_concat(self, indices_or_sections, split_axis=0, concat_axis=1) -> Dataset:
        def split_concat(v):
            return jnp.concatenate(
                jnp.split(v, indices_or_sections, axis=split_axis),
                axis=concat_axis,
            )

        return tree.map(split_concat, self)

    def map(self, fn: Callable[[Array], Any]) -> Dataset:
        return tree.map(fn, self)

    def fit_logits(self) -> Dataset:
        from neuralab.trading.evaluation import fit_target_logits

        target_logits = fit_target_logits(self)
        return self.replace(logits=target_logits)

    @cached_property
    def trends(self):
        from neuralab.trading.trends import Trends

        return Trends.from_dataset(self)


cli = Typer(name="dataset")


@cli.command("ensure")
def ensure(
    time_range: Annotated[TimeRange, Argument(parser=TimeRange.of)],
    # symbolset: Annotated[Symbol, Argument(parser=str)],
    # marketset: Annotated[Market, Argument(parser=str)],
    fit_logits: Annotated[bool, Option()] = True,
    local: Annotated[bool, Option()] = True,
    remote: Annotated[bool, Option()] = False,
    checking: Annotated[bool, Option()] = True,
):
    """Ensure the availability of the training dataset"""

    Dataset.Ref(
        time_range=time_range,
        symbol_set=DEFAULT_SYMBOLSET,
        market_set=DEFAULT_MARKETSET,
    ).ensure(local=local, remote=remote, checking=checking)


# %%
if __name__ == "__main__":
    ds = Dataset.Ref.of("2024 1m").fetch()
