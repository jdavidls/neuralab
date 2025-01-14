# %%
from __future__ import annotations

import gc
from abc import ABC
from dataclasses import dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import IO, Annotated, Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch as pt
from typer import Argument, Option, Typer
from pandas import DataFrame, read_parquet

from neuralab import track
from neuralab.resource import Resource, Url, ensure_tree, fetch_tree
from neuralab.trading.knowledge import DEFAULT_MARKETSET, DEFAULT_SYMBOLSET, Market, Symbol
from neuralab.utils.timeutils import Date, TimeDelta, TimeRange, milliseconds

log = get_logger(__name__)

type Disposition = Literal["sampled", "full"]

COMPRESSION_FORMAT = "gzip"  # {‘snappy’, ‘gzip’, ‘brotli’, ‘lz4’, ‘zstd’}


@dataclass(frozen=True)
class DataframeResource(Resource.Ref[DataFrame], ABC):
    symbol: Symbol
    market: Market

    @property
    def path_suffix(self):
        return f".parquet.{COMPRESSION_FORMAT}"

    @property
    def type(self):
        return DataFrame

    def load(self, buffer: IO[bytes]):
        return read_parquet(buffer)

    def dump(self, buffer: IO[bytes], resource: DataFrame):
        resource.to_parquet(buffer, compression=COMPRESSION_FORMAT)


@dataclass(frozen=True)
class TradeDataFrame(DataframeResource):
    date: Date

    @classmethod
    def for_time_range(
        cls,
        symbol: Symbol,
        market: Market,
        time_range: TimeRange,
    ):
        return [
            TradeDataFrame(symbol, market, day)
            for day in time_range.replace(step=TimeDelta(days=1))
        ]

    @property
    def path(self):
        return (
            Path("daily-trade")
            / f"{self.market}-{self.symbol.lower()}"
            / self.date.isoformat()
        )

    @property
    def url(self) -> Url:
        match self.market:
            case "binance-spot":
                return Url(
                    f"https://data.binance.vision/data/spot/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{self.date}.zip"
                )
            case "binance-usdtm":
                return Url(
                    f"https://data.binance.vision/data/futures/um/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{self.date}.zip"
                )
            case "binance-coinm":
                return Url(
                    f"https://data.binance.vision/data/futures/cm/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{self.date}.zip"
                )
            case _:
                raise ValueError(f"Invalid market {self.market}")

    @property
    def column_info(self):
        match self.market:
            case "binance-spot":
                return {
                    "id": dict(dtype=np.int64),
                    "price": dict(dtype=np.float32),
                    "qty": dict(dtype=np.float32),
                    "initial_trade_id": dict(dtype=np.int64),
                    "final_trade_id": dict(dtype=np.int64),
                    "time": dict(dtype=np.int64),
                    "is_bid": dict(dtype=np.bool_),
                    "best_match": dict(dtype=np.bool_),
                }
            case "binance-usdtm" | "binance-coinm":
                return {
                    "agg_trade_id": dict(dtype=np.int64, rename="id"),
                    "price": dict(dtype=np.float32),
                    "quantity": dict(dtype=np.float32, rename="qty"),
                    "first_trade_id": dict(dtype=np.int64, rename="initial_trade_id"),
                    "last_trade_id": dict(dtype=np.int64, rename="final_trade_id"),
                    "transact_time": dict(dtype=np.int64, rename="time"),
                    "is_buyer_maker": dict(dtype=np.bool_, rename="is_bid"),
                }
            case _:
                raise ValueError(f"Invalid market {self.market}")

    @property
    def column_names(self):
        return list(self.column_info.keys())

    @property
    def column_dtypes(self):
        return {k: v["dtype"] for k, v in self.column_info.items()}

    @property
    def column_renames(self):
        return {k: v["rename"] for k, v in self.column_info.items() if "rename" in v}

    @property
    def has_head_line(self):
        match self.market:
            case "binance-spot":
                return False
            case "binance-usdtm" | "binance-coinm":
                return True

    def make(self):
        with track.task(
            total=None,
            description=f"Reading CSV {self.url.url}",
        ) as task:
            try:
                with ZipFile(self.url.download()) as zip_file:
                    csv_filename = f"{self.symbol}-aggTrades-{self.date}.csv"
                    # task.update(total=zip_file.getinfo(csv_filename).file_size)

                    with zip_file.open(csv_filename) as csv:
                        return pd.read_csv(
                            # task.wrap_io("read", csv),
                            csv,
                            names=self.column_names,
                            dtype=self.column_dtypes,
                            # index_col=self.column_names[0],
                            skiprows=1 if self.has_head_line else 0,
                        ).rename(columns=self.column_renames, errors="raise")
            except Exception as e:
                e.add_note(f"Error reading {self.url.url}")
                task.console.log(f"[red]Error reading CSV[/red] {self.url.url}")
                raise e


@dataclass(frozen=True)
class TradeSampleDataFrame(DataframeResource):
    time_range: TimeRange

    @property
    def path(self):
        return (
            Path("sampled")
            / f"{self.market}-{self.symbol.lower()}"
            / f"{self.time_range}"
        )

    def make(self):
        assert self.time_range.is_for_exact_days

        if day := self.time_range.is_for_an_exact_day:
            trade_df = TradeDataFrame(self.symbol, self.market, day).fetch()
            sample_df = sample_trade_dataframe(trade_df, self.time_range)
            del trade_df
            gc.collect()
            return sample_df

        daily_dataframes = fetch_tree(
            [
                TradeSampleDataFrame(self.symbol, self.market, day_range)
                for day_range in self.time_range.each(TimeDelta(days=1))
            ],
            description=f"Making {self.path}",
        )

        return pd.concat(daily_dataframes)


def sample_trade_dataframe(
    full_df: DataFrame,
    time_range: TimeRange,
):

    time = pt.tensor(full_df["time"].to_numpy(), dtype=pt.long)
    price = pt.tensor(full_df["price"].to_numpy(), dtype=pt.float32)
    qty = pt.tensor(full_df["qty"].to_numpy(), dtype=pt.float32)
    is_bid = pt.tensor(full_df["is_bid"].to_numpy(), dtype=pt.bool)

    is_ask = ~is_bid

    indices = (time - milliseconds(time_range.start)) // milliseconds(time_range.step)

    def sample_qty(indices, qty):
        return pt.zeros(len(time_range), dtype=pt.float32).scatter_add_(0, indices, qty)

    def sample_price(indices, price, qty, sample_high_and_low=True):

        events, inv = indices.unique(return_inverse=True)

        _, fix = (
            pt.zeros(len(time_range), dtype=pt.long)
            .scatter_(0, events, events.diff(prepend=events[0:1]))
            .cumsum_(dim=0)
            .unique(return_inverse=True)
        )

        vwap = (
            pt.zeros_like(events, dtype=pt.float32).scatter_add_(0, inv, price * qty)
        ) / (pt.zeros_like(events, dtype=pt.float32).scatter_add_(0, inv, qty))

        vwap = vwap[fix]

        ## TODO: calcular la varianza o desviacion estandard de los precios respecto de vwap

        if not sample_high_and_low:
            return vwap

        high = pt.zeros_like(events, dtype=pt.float32).scatter_reduce_(
            0, inv, price, "amax", include_self=False
        )

        low = pt.zeros_like(events, dtype=pt.float32).scatter_reduce_(
            0, inv, price, "amin", include_self=False
        )

        high = high[fix]
        low = low[fix]

        return vwap, high, low

    vol = sample_qty(indices, qty)
    bid_vol = sample_qty(indices[is_bid], qty[is_bid])
    ask_vol = sample_qty(indices[is_ask], qty[is_ask])

    vwap, high, low = sample_price(indices, price, qty)
    # bid_vwap = sample_price(indices[is_bid], price[is_bid], qty[is_bid], False)
    # ask_vwap = sample_price(indices[is_ask], price[is_ask], qty[is_ask], False)

    return pd.DataFrame.from_dict(
        {
            "high": high,
            "low": low,
            "vwap": vwap,
            # 'bid_vwap': bid_vwap,
            # 'ask_vwap': ask_vwap,
            "vol": vol,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
        }
    )


cli = Typer(name="dataframe")


@cli.command("ensure")
def ensure(
    time_range: Annotated[TimeRange, Argument(parser=TimeRange.of)],
    # symbolset: Annotated[Symbol, Argument(parser=str)],
    # marketset: Annotated[Market, Argument(parser=str)],
    disposition: Annotated[Disposition, Option(parser=str)] = "sampled",
    local: Annotated[bool, Option()] = True,
    remote: Annotated[bool, Option()] = False,
    checking: Annotated[bool, Option()] = True,
):
    """Ensure the availability of the trade dataframes for the given symbol and market."""

    symbols = DEFAULT_SYMBOLSET
    markets = DEFAULT_MARKETSET

    match disposition:
        case "full":
            ensure_tree(
                [
                    TradeDataFrame(symbol, market, day)
                    for market in markets
                    for symbol in symbols
                    for day in time_range.replace(step=TimeDelta(days=1))
                ],
                local=local,
                remote=remote,
                checking=checking,
            )
        case "sampled":
            ensure_tree(
                [
                    TradeSampleDataFrame(symbol, market, time_range)
                    for market in markets
                    for symbol in symbols
                ],
                local=local,
                remote=remote,
                checking=checking,
            )
        case _:
            raise ValueError(f"Invalid disposition {disposition}")


# %%
if __name__ == "__main__":

    ensure(TimeRange.of("2023-01-01_1m"), checking=True)

    # TradeSampleDataFrame('BTCUSDT', 'binance-usdtm', TimeRange.of("2023-01-01 1m")).fetch()

    # # %%
    # with ZipFile(tdf.url.download()) as zipfile:
    #     print(zipfile.namelist())
    #     with zipfile.open(zipfile.namelist()[0]) as csv:
    #         print(pd.read_csv(csv, nrows=10))
