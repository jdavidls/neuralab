# %%
from __future__ import annotations

from io import BytesIO
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal, NamedTuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch as pt

from dataproxy.loader import Loader
from timeutils import Date, TimeRange, millis

log = get_logger(__name__)

type Market = Literal["binance-spot", "binance-usdtm", "binance-coinm"]
type Symbol = str
type Disposition = Literal["sampled", "full"]

DEFAULT_SYMBOLS: list[Symbol] = ["BTCUSDT", "ETHUSDT"]
DEFAULT_MARKETS: list[Market] = ["binance-spot", "binance-usdtm"]

DEFAULT_SAMPLE_RATE = "1m"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_STOP_DATE = "2024-01-01"
DEFAULT_TIME_RANGE = TimeRange.of(
    DEFAULT_START_DATE, DEFAULT_STOP_DATE, DEFAULT_SAMPLE_RATE
)

_STORAGE_PATH = Path.home() / ".dataproxy"

_COMPRESSION_FORMAT = "gzip"  # {‘snappy’, ‘gzip’, ‘brotli’, ‘lz4’, ‘zstd’}

_TRADE_SOURCE_URL = {
    "binance-spot": lambda ref: f"https://data.binance.vision/data/spot/daily/aggTrades/{ref.symbol}/{ref.symbol}-aggTrades-{ref.date}.zip",
    "binance-usdtm": lambda ref: f"https://data.binance.vision/data/futures/um/daily/aggTrades/{ref.symbol}/{ref.symbol}-aggTrades-{ref.date}.zip",
    "binance-coinm": lambda ref: f"https://data.binance.vision/data/futures/cm/daily/aggTrades/{ref.symbol}/{ref.symbol}-aggTrades-{ref.date}.zip",
}

_BINANCE_SPOT_COL_DTYPES = {
    "id": np.int64,
    "price": np.float32,
    "qty": np.float32,
    "initial_trade_id": np.int64,
    "final_trade_id": np.int64,
    "time": np.int64,
    "is_bid": np.bool_,
    "best_match": np.bool_,
}

_BINANCE_FUT_COL_DTYPES = {
    "agg_trade_id": np.int64,
    "price": np.float32,
    "quantity": np.float32,
    "first_trade_id": np.int64,
    "last_trade_id": np.int64,
    "transact_time": np.int64,
    "is_buyer_maker": np.bool_,
}

_BINANCE_FUT_COL_RENAMES = {
    "agg_trade_id": "id",
    "quantity": "qty",
    "first_trade_id": "initial_trade_id",
    "last_trade_id": "final_trade_id",
    "transact_time": "time",
    "is_buyer_maker": "is_bid",
}


def binance_spot_loader(ref: DataFrameRef, content: bytes):
    with ZipFile(BytesIO(content)) as zip_file:
        with zip_file.open(f"{ref.symbol}-aggTrades-{ref.date}.csv") as csv:
            return pd.read_csv(
                csv,
                names=list(_BINANCE_SPOT_COL_DTYPES.keys()),
                dtype=_BINANCE_SPOT_COL_DTYPES,
                index_col="id",
                skiprows=1,
            )


def binance_fut_loader(ref: DataFrameRef, content: bytes):
    with ZipFile(BytesIO(content)) as zip_file:
        with zip_file.open(f"{ref.symbol}-aggTrades-{ref.date}.csv") as csv:
            return pd.read_csv(
                csv,
                names=list(_BINANCE_FUT_COL_DTYPES.keys()),
                dtype=_BINANCE_FUT_COL_DTYPES,
                index_col="agg_trade_id",
                skiprows=1,
            ).rename(_BINANCE_FUT_COL_RENAMES)


_LOADERS = {
    "binance-spot": binance_spot_loader,
    "binance-usdtm": binance_fut_loader,
    "binance-coinm": binance_fut_loader,
}


class DataFrameRef(NamedTuple):
    symbol: Symbol
    market: Market
    date: Date

    @property
    def remote_url(self):
        return _TRADE_SOURCE_URL[self.market](self)

    def local_path(self, disposition: Disposition = "sampled"):
        return (
            _STORAGE_PATH
            / disposition
            / f"{self.market}-{self.symbol.lower()}"
            / self.date.isoformat()
        ).with_suffix(f".parquet.{_COMPRESSION_FORMAT}")

    @property
    def loader(self):
        return _LOADERS[self.market]



def fetch_dataframes(
    time_range: TimeRange,
    symbols: list[Symbol] = DEFAULT_SYMBOLS,
    markets: list[Market] = DEFAULT_MARKETS,
    *,
    disposition: Disposition = "sampled",
    only_ensure=False,
    num_workers=8,
):

    tasks = [
        DataFrameRef(sym, market, dt)
        for dt in time_range.dates()
        for sym in symbols
        for market in markets
    ]

    with Loader.pool(f"Fetching datasets", max_workers=num_workers) as loader:
        for ref in tasks:

            def ensure_full_dataframe(ref: DataFrameRef):

                local_path = ref.local_path("full")

                if local_path.exists():
                    try:
                        return pd.read_parquet(local_path)
                    except Exception as e:
                        log.warning(f"Failed reading {local_path}")
                        pass

                content = loader.download(
                    ref.remote_url, title=f"{ref.symbol} @ {ref.date} {ref.market}"
                )

                try:
                    full_df = ref.loader(ref, content)
                except Exception as e:
                    e.add_note(f"Failed fetching {ref.remote_url}")
                    raise

                full_df = full_df[["time", "price", "qty", "is_bid"]]

                local_path.parent.mkdir(parents=True, exist_ok=True)
                full_df.to_parquet(local_path, compression=_COMPRESSION_FORMAT)

                return full_df

            def ensure_sampled_dataframe(ref: DataFrameRef):

                local_path = ref.local_path("sampled")

                if local_path.exists():
                    try:
                        return pd.read_parquet(local_path)
                    except Exception as e:
                        log.warning(f"Failed reading {local_path}")
                        pass

                full_df = ensure_full_dataframe(ref)
                smp_df = sample_full_dataframe(
                    full_df, TimeRange.from_date(ref.date, time_range.step)
                )

                local_path.parent.mkdir(parents=True, exist_ok=True)
                smp_df.to_parquet(local_path, compression=_COMPRESSION_FORMAT)

                return smp_df

            def wrap_task(task, ref):
                res = task(ref)

                if only_ensure:
                    res = None  # libera memoria
                return ref, res

            match disposition:
                case "sampled":
                    loader.add_task(wrap_task, ensure_sampled_dataframe, ref)
                case "full":
                    loader.add_task(wrap_task, ensure_full_dataframe, ref)

    results = {ref: df for ref, df in loader.get_results()}

    if only_ensure:
        return
    
    res = lambda sym, mkt, dt: results[DataFrameRef(sym, mkt, dt)]
    cat = lambda sym, mkt: pd.concat([ res(sym, mkt, dt) for dt in time_range.dates() ])
    mkt = lambda sym: { mkt: cat(sym, mkt) for mkt in markets }
    return { sym: mkt(sym) for sym in symbols }



def sample_full_dataframe(
    full_df: pd.DataFrame,
    time_range: TimeRange,
):

    time = pt.tensor(full_df["time"].to_numpy(), dtype=pt.long)
    price = pt.tensor(full_df["price"].to_numpy(), dtype=pt.float32)
    qty = pt.tensor(full_df["qty"].to_numpy(), dtype=pt.float32)
    is_bid = pt.tensor(full_df["is_bid"].to_numpy(), dtype=pt.bool)
    is_ask = ~is_bid

    indices = (time - millis(time_range.start)) // millis(time_range.step)

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


# %%
if __name__ == "__main__":
    from treescope import display
    tr = TimeRange.of("2023-01-01", "2024-01-01", "1m")
    dfs = fetch_dataframes(tr, disposition="sampled")

    display(dfs)
