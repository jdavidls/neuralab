# %%
from __future__ import annotations

import pickle
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Optional
from weakref import WeakValueDictionary

import optax
from flax import nnx, struct
from jax import numpy as jnp
from jax import random, tree
from jaxtyping import Array, Float, Int
from pandas import DataFrame
from tqdm.notebook import tqdm, trange

from neuralab.trading.dataframe import (
    DEFAULT_MARKETS,
    DEFAULT_SYMBOLS,
    DEFAULT_TIME_RANGE,
    Market,
    Symbol,
    fetch_dataframes,
)
from neuralab.utils.timeutils import TimeRange

NL_HOME_PATH = Path.home() / ".neuralab"

type Feature = Literal[
    "log_price", "returns", "diff_log_price", "log_volume", "log_imbalance"
]

_loaded_datasets: WeakValueDictionary[str, Dataset] = WeakValueDictionary()


def last_power_of_two(x: int):
    return 2 ** int(jnp.log2(x))


_DEFAULTS = {
    "default": lambda: Dataset.fetch().with_weights,
    "default-fit": lambda: (
        Dataset.load("default").split_concat(32)[: 2**14].split_concat(8).with_trends
    ),
}


@struct.dataclass
class Dataset(struct.PyTreeNode):  # un dataset es un modulo porque tiene pesos!!
    log_price: jnp.ndarray
    returns: jnp.ndarray
    diff_log_price: jnp.ndarray
    log_volume: jnp.ndarray
    log_imbalance: jnp.ndarray

    weights: Optional[jnp.ndarray] = None  # weights

    @property
    def shape(self):
        return self.log_price.shape

    @cached_property
    def with_weights(self):  # TODO: tags seran renombrados por weights
        return self if self.weights is not None else fit_weights(self)

    @property
    def with_trends(self):  # TODO: tags seran renombrados por weights
        _ = self.trends
        return self

    @cached_property
    def trends(self) -> Trends:
        return Trends.from_dataset(self.with_weights)

    def __len__(self):
        return len(self.log_price)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, args) -> Dataset:
        return tree.map(lambda v: v.__getitem__(args), self)

    def features(self, feature_names: list[Feature], axis=-1):
        return jnp.stack(
            [getattr(self, feature) for feature in feature_names], axis=axis
        )

    @staticmethod
    def concat(datasets: list[Dataset], axis=1):
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

    @classmethod
    def from_dataframe(cls, df: DataFrame):

        def df_timeserie(nm: str):
            return df[nm].to_numpy()[:, None, None]

        price = df_timeserie("vwap")
        vol = df_timeserie("vol")
        ask_vol = df_timeserie("ask_vol")
        bid_vol = df_timeserie("bid_vol")

        log_price = jnp.log(price)

        return cls(
            log_price=log_price,
            returns=jnp.diff(log_price, append=log_price[-1:], axis=0),
            diff_log_price=jnp.diff(log_price, prepend=log_price[:1], axis=0),
            log_volume=jnp.log(vol),
            log_imbalance=jnp.log(ask_vol) - jnp.log(bid_vol),
        )

    @classmethod
    def fetch(
        cls,
        time_range: TimeRange = DEFAULT_TIME_RANGE,
        symbols: list[Symbol] = DEFAULT_SYMBOLS,
        markets: list[Market] = DEFAULT_MARKETS,
    ):
        dfs = fetch_dataframes(time_range, symbols, markets)

        return cls.concat(
            [
                cls.concat(
                    [cls.from_dataframe(df) for df in df_by_mkt.values()], axis=1
                )
                for df_by_mkt in dfs.values()
            ],
            axis=2,
        )

    def save(self, name: str):
        global _loaded_datasets

        path = NL_HOME_PATH / type(self).__qualname__
        path.mkdir(parents=True, exist_ok=True)

        with open(path / f"{name}.pkl", "wb") as f:
            pickle.dump(self, f)

        _loaded_datasets[name] = self

    @classmethod
    def load(cls, name: str = "default") -> Dataset:
        global _loaded_datasets

        if name in _loaded_datasets:
            return _loaded_datasets[name]

        filename = NL_HOME_PATH / cls.__qualname__ / f"{name}.pkl"

        try:
            with filename.open("rb") as f:
                dataset = pickle.load(f)
                _loaded_datasets[name] = dataset
                return dataset

        except FileNotFoundError:
            if name not in _DEFAULTS:
                raise

        ds_builder = _DEFAULTS[name]
        dataset = ds_builder()
        dataset.save(name)
        return dataset


@struct.dataclass
class Trends(struct.PyTreeNode):
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
    def duration(self):
        return self.stop_at - self.start_at

    @cached_property
    def start_log_price(self):
        return self.dataset.log_price[self.start_at, self.asset, self.market]

    @cached_property
    def stop_log_price(self):
        return self.dataset.log_price[self.stop_at, self.asset, self.market]

    @cached_property
    def returns(self):
        return jnp.abs(self.stop_log_price - self.start_log_price)

    @cached_property
    def direction(self):
        return jnp.sign(self.stop_log_price - self.start_log_price)

    def __len__(self):
        return len(self.start_at)

    def __getitem__(self, *args) -> Trends:
        return tree.map(lambda v: v.__getitem__(*args), self)

    @staticmethod
    def concat(datasets: list[Trends]):
        return tree.map(lambda *v: jnp.concat(v, axis=0), *datasets)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Trends:
        assert dataset.weights is not None
        events = jnp.diff(
            jnp.sign(dataset.weights), axis=0, append=dataset.weights[-1:]
        )

        t_idx, a_idx, m_idx = jnp.nonzero(events)

        def trend_fn(a, m):
            start_idx = t_idx[(a_idx == a) & (m_idx == m)]
            start_idx, stop_idx = start_idx[:-1], start_idx[1:]
            return Trends(
                dataset=dataset,
                asset=jnp.full_like(stop_idx, a),
                market=jnp.full_like(stop_idx, m),
                start_at=start_idx,
                stop_at=stop_idx,
            )

        am = [
            (a, m)
            for a in range(dataset.weights.shape[1])
            for m in range(dataset.weights.shape[2])
        ]

        return Trends.concat([trend_fn(a, m) for a, m in tqdm(am, "Computing trends")])


def fit_weights(
    dataset,
    weights: Optional[jnp.ndarray] = None,
    epochs=1000,
    lr=1e-2,
    mode="max_total_perf",
):
    from neuralab.trading.sim import SimParams, sim

    weights = dataset.returns

    opt = optax.nadam(lr)
    opt_state = opt.init(weights)
    sim_params = SimParams(transaction_cost=0.001)

    fit_activation = nnx.hard_sigmoid
    eval_activation = lambda x: nnx.relu(jnp.sign(x))

    @nnx.jit
    def fit_step(opt_state, weights, dataset):

        def loss_fn(weights):
            return sim(dataset, fit_activation(weights), params=sim_params).loss(mode)

        loss, grads = nnx.value_and_grad(loss_fn)(weights)
        updates, opt_state = opt.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)

        return opt_state, weights, loss, jnp.std(grads)

    try:
        tqdm = trange(epochs)
        for n in tqdm:
            if n % 25 == 0:
                s = sim(dataset, eval_activation(weights))
                perf = jnp.mean(s.total_performance)
                cost = jnp.mean(s.total_transaction_cost)
                turnover = jnp.mean(s.total_turnover)

            opt_state, weights, loss, g_std = fit_step(opt_state, weights, dataset)
            w_std = jnp.std(weights)

            tqdm.set_description(
                f"Gain: {-loss:.2f} "
                f"Perf: {perf:.2f} "
                f"Cost: {cost:.2f} "
                f"Turnover: {turnover:.0f} "
                f"Weights: ±{w_std:.2f}"
            )

    except KeyboardInterrupt:
        pass

    return dataset.replace(weights=weights)


if __name__ == "__main__":
    ds = Dataset.load()
