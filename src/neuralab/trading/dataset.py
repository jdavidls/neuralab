# %%
from __future__ import annotations

import pickle
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional

import optax
from flax import nnx, struct
from jax import numpy as jnp
from jax import random, tree
from jaxtyping import Array, Float, Int
from pandas import DataFrame
from tqdm import tqdm, trange

from neuralab.trading.dataframe import (
    DEFAULT_MARKETS,
    DEFAULT_SYMBOLS,
    DEFAULT_TIME_RANGE,
    Market,
    Symbol,
    fetch_dataframes,
)
from neuralab.utils.timeutils import TimeRange

NL_TRADING_PATH = Path.home() / ".neuralab" / "trading"
NL_TRADING_PATH.mkdir(parents=True, exist_ok=True)

NL_DEFAULT_DATASET_PATH = NL_TRADING_PATH / "default_dataset.pkl"

type Feature = Literal[
    "log_price", "returns", "diff_log_price", "log_volume", "log_imbalance"
]
_DEFAULT = None


@struct.dataclass
class Dataset(struct.PyTreeNode):
    log_price: jnp.ndarray
    returns: jnp.ndarray
    diff_log_price: jnp.ndarray
    log_volume: jnp.ndarray
    log_imbalance: jnp.ndarray
    tags: Optional[jnp.ndarray] = None

    @property
    def shape(self):
        return self.log_price.shape

    @cached_property
    def with_tags(self):
        return self if self.tags is not None else fit_tags(self)

    @cached_property
    def trends(self):
        return Trends.from_dataset(self.with_tags)

    def __len__(self):
        return len(self.log_price)

    def __getitem__(self, *args):
        return tree.map(lambda v: v.__getitem__(*args), self)

    def features(self, feature_names: list[Feature], axis=-1):
        return jnp.stack(
            [getattr(self, feature) for feature in feature_names], axis=axis
        )

    @staticmethod
    def stack(datasets: list[Dataset], axis=0):
        return tree.map(lambda *v: jnp.stack(v, axis=axis), *datasets)

    def split(self, indices_or_sections, axis=0):
        return tree.map(lambda v: jnp.split(v, indices_or_sections, axis=axis), self)

    def split_concat(self, indices_or_sections, split_axis=0, concat_axis=1):
        def split_concat(v):
            return jnp.concatenate(
                jnp.split(v, indices_or_sections, axis=split_axis),
                axis=concat_axis,
            )

        return tree.map(split_concat, self)

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
        if _DEFAULT is not None:
            return _DEFAULT

        if NL_DEFAULT_DATASET_PATH.exists():
            with open(NL_DEFAULT_DATASET_PATH, "rb") as f:
                dataset = pickle.load(f)
        else:
            dataset = Dataset.fetch().with_tags.split_concat(100)
            _ = dataset.trends

            with open(NL_DEFAULT_DATASET_PATH, "wb") as f:
                pickle.dump(dataset, f)

        _DEFAULT = dataset
        return dataset


@struct.dataclass
class Trends(struct.PyTreeNode):
    dataset: Dataset = struct.field(pytree_node=False)

    asset: Int[Array, "trends"]
    market: Int[Array, "trends"]
    start_at: Int[Array, "trends"]
    stop_at: Int[Array, "trends"]

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
        return self.stop_log_price - self.start_log_price

    @cached_property
    def direction(self):
        return jnp.sign(self.returns)

    def __len__(self):
        return len(self.start_at)

    def __getitem__(self, *args):
        return tree.map(lambda v: v.__getitem__(*args), self)

    @staticmethod
    def concat(datasets: list[Trends]):
        return tree.map(lambda *v: jnp.concat(v, axis=0), *datasets)

    @classmethod
    def from_dataset(cls, dataset: Dataset):
        assert dataset.tags is not None
        events = jnp.diff(jnp.sign(dataset.tags), axis=0, append=dataset.tags[-1:])

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
            for a in range(dataset.tags.shape[1])
            for m in range(dataset.tags.shape[2])
        ]

        return Trends.concat([trend_fn(a, m) for a, m in tqdm(am, "Building trends")])


def fit_tags(
    dataset,
    tags: Optional[jnp.ndarray] = None,
    epochs=1000,
    lr=1e-2,
    mode="max_total_perf",
):
    from neuralab.trading.sim import SimParams, sim

    tags = dataset.returns

    opt = optax.nadam(lr)
    opt_state = opt.init(tags)
    sim_params = SimParams(transaction_cost=0.001)

    fit_activation = nnx.hard_sigmoid
    eval_activation = lambda x: nnx.relu(jnp.sign(x))

    @nnx.jit
    def fit_step(opt_state, labels, dataset):

        def loss_fn(labels):
            return sim(dataset, fit_activation(labels), params=sim_params).loss(mode)

        loss, grads = nnx.value_and_grad(loss_fn)(labels)
        updates, opt_state = opt.update(grads, opt_state)
        labels = optax.apply_updates(labels, updates)

        return opt_state, labels, loss, jnp.std(grads)

    try:
        tqdm = trange(epochs)
        for n in tqdm:
            if n % 25 == 0:
                s = sim(dataset, eval_activation(tags))
                perf = jnp.mean(s.total_performance)
                cost = jnp.mean(s.total_transaction_cost)
                turnover = jnp.mean(s.total_turnover)

            opt_state, tags, mode, g_std = fit_step(opt_state, tags, dataset)
            l_std = jnp.std(tags)

            tqdm.set_description(
                f"Gain: {-mode:.2f} "
                f"Perf: {perf:.2f} "
                f"Cost: {cost:.2f} "
                f"Turn: {turnover:.2f} "
                f"Labels: ±{l_std:.2f}"
            )

    except KeyboardInterrupt:
        pass

    return dataset.replace(tags=tags)
