# %%
from __future__ import annotations
from typing import Literal

from flax import struct
from jax import numpy as jnp
from jax import tree
from pandas import DataFrame

from timeutils import TimeRange

from dataproxy.dataframe import (
    DEFAULT_MARKETS,
    DEFAULT_SYMBOLS,
    DEFAULT_TIME_RANGE,
    Market,
    Symbol,
    fetch_dataframes,
)

type Feature = Literal["log_price", "returns", "diff_log_price", "log_volume", "log_imbalance"]
_DEFAULT = None

@struct.dataclass
class Dataset(struct.PyTreeNode):
    log_price: jnp.ndarray
    returns: jnp.ndarray
    diff_log_price: jnp.ndarray
    log_volume: jnp.ndarray
    log_imbalance: jnp.ndarray

    def __len__(self):
        return len(self.log_price)

    def __getitem__(self, *args):
        return tree.map(lambda v: v.__getitem__(*args), self)

    @property
    def shape(self):
        return tree.map(lambda v: v.shape, self)

    @property
    def dtype(self):
        return tree.map(lambda v: v.dtype, self)


    @staticmethod
    def stack(datasets: list[Dataset], axis=0):
        return tree.map(lambda *v: jnp.stack(v, axis=axis), *datasets)

    def split(self, indices_or_sections, axis=0):
        return tree.map(lambda v: jnp.split(v, indices_or_sections, axis=axis), self)


    @classmethod
    def from_dataframe(cls, df: DataFrame):
        log_price = jnp.log(df["vwap"].to_numpy())
        diff_log_price = jnp.diff(log_price, prepend=log_price[0])
        returns = jnp.diff(log_price, append=log_price[-1])

        log_volume = jnp.log(df["vol"].to_numpy())
        log_imbalance = jnp.log(df["ask_vol"].to_numpy()) - jnp.log(
            df["bid_vol"].to_numpy()
        )

        return cls(
            log_price=log_price,
            returns=returns,
            diff_log_price=diff_log_price,
            log_volume=log_volume,
            log_imbalance=log_imbalance,
        )

    @classmethod
    def fetch(
        cls,
        time_range: TimeRange = DEFAULT_TIME_RANGE,
        symbols: list[Symbol] = DEFAULT_SYMBOLS,
        markets: list[Market] = DEFAULT_MARKETS,
    ):
        dfs = fetch_dataframes(time_range, symbols, markets)

        return cls.stack(
            [
                cls.stack([cls.from_dataframe(df) for df in df_by_mkt.values()], axis=1)
                for df_by_mkt in dfs.values()
            ],
            axis=1,
        )
    
    @classmethod
    def default(cls):
        global _DEFAULT
        if _DEFAULT is None:
            _DEFAULT = Dataset.fetch()
        return _DEFAULT

    def features(self, feature_names: list[Feature], axis=-1):
        return jnp.stack([getattr(self, feature) for feature in feature_names], axis=axis)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(self.log_price)
        plt.show()

