#%%

from io import BytesIO
from logging import getLogger as get_logger
from pathlib import Path
from random import sample
from typing import Optional
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch as pt

from dataproxy.loader import Loader
from timeutils import Date, TimeDelta, TimeRange, millis

DATAPROXY_STORAGE_PATH = Path.home() / '.dataproxy'

BINANCE_STORAGE_PATH = DATAPROXY_STORAGE_PATH / 'binance'
BINANCE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


COMPRESSION = 'gzip' # {‘snappy’, ‘gzip’, ‘brotli’, ‘lz4’, ‘zstd’}


log = get_logger(__name__)

_AGG_TRADE_COLUMN_DTYPE = {
    "id": np.int64,
    "price": np.float32,
    "qty": np.float32,
    "initial_trade_id": np.int64,
    "final_trade_id": np.int64,
    "time": np.int64,
    "is_bid": np.bool_,
    "best_match": np.bool_,
}

_AGG_TRADE_COLUMN_NAMES = list(_AGG_TRADE_COLUMN_DTYPE.keys())

def _storage_path(date: Date, symbol: str):
    return (BINANCE_STORAGE_PATH / symbol / date.isoformat()).with_suffix(f'.parquet.{COMPRESSION}')

def _agg_trade_url(date: Date, symbol: str):
    return f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"

def fetch_trade_dataframe(time_range: TimeRange, symbol: str):
    with Loader.pool(f"Fetching {symbol} @ {time_range}", max_workers=8) as loader:
        for dt in time_range.dates():
            @loader.task(dt)
            def task(dt):
                storage_path = _storage_path(dt, symbol)

                if storage_path.exists():
                    return pd.read_parquet(storage_path)

                url = _agg_trade_url(dt, symbol)
                content = loader.download(url)

                with ZipFile(BytesIO(content)) as zip_file:
                    try:
                        with zip_file.open(f"{symbol}-aggTrades-{dt}.csv") as csv:
                            df = pd.read_csv(
                                csv, 
                                names=_AGG_TRADE_COLUMN_NAMES, 
                                dtype=_AGG_TRADE_COLUMN_DTYPE,
                                index_col="id"
                            )
                    except:
                        print(zip_file.namelist(), dt, url)
                        raise

                df = df[["time", "price", "qty", "is_bid"]]

                storage_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(storage_path, compression='gzip')

                return df

    return pd.concat(loader.results)

#%%

def process_sample_dataframe(
    df: pd.DataFrame, 
    time_range: TimeRange, 
):   

    time = pt.tensor(df['time'].to_numpy(), dtype=pt.long)
    price = pt.tensor(df['price'].to_numpy(), dtype=pt.float32)
    qty = pt.tensor(df['qty'].to_numpy(), dtype=pt.float32)
    is_bid = pt.tensor(df['is_bid'].to_numpy(), dtype=pt.bool)
    is_ask = ~is_bid

    indices = (time - millis(time_range.start)) // millis(time_range.step)


    def sample_qty(indices, qty):
        return (
            pt.zeros(len(time_range), dtype=pt.float32)
              .scatter_add_(0, indices, qty)
        )

    def sample_price(indices, price, qty, sample_high_and_low=True):

        events, inv = indices.unique(return_inverse=True)

        _, fix = (
            pt.zeros(len(time_range), dtype=pt.long)
            .scatter_(0, events, events.diff(prepend=events[0:1]))
            .cumsum_(dim=0)
            .unique(return_inverse=True)
        )

        vwap = (
            pt.zeros_like(events, dtype=pt.float32)
            .scatter_add_(0, inv, price * qty)
        ) / (
            pt.zeros_like(events, dtype=pt.float32)
            .scatter_add_(0, inv, qty)
        )

        vwap = vwap[fix]

        if not sample_high_and_low:
            return vwap

        high = (
            pt.zeros_like(events, dtype=pt.float32)
            .scatter_reduce_(0, inv, price, 'amax', include_self=False)
        )

        low = (
            pt.zeros_like(events, dtype=pt.float32)
            .scatter_reduce_(0, inv, price, 'amin', include_self=False)
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

    return pd.DataFrame.from_dict({
        'high': high,
        'low': low,
        'vwap': vwap,
        # 'bid_vwap': bid_vwap,
        # 'ask_vwap': ask_vwap,
        'vol': vol,
        'bid_vol': bid_vol,
        'ask_vol': ask_vol
    })



#%%
if __name__ == '__main__':
    
    tr = TimeRange.of(
        '2021-04-01', 
        '2021-05-01', 
        TimeDelta(hours=4)
    )

    tdf = fetch_trade_dataframe(tr, "ETHUSDT")
    #%%
    sdf = process_sample_dataframe(tdf, tr)
    #%%
    from matplotlib import pyplot as plt
    
    sdf_ = sdf
    plt.plot(sdf_['vwap'], label='vwap')
    plt.plot(sdf_['high'], label='high')
    plt.plot(sdf_['low'], label='low')
    plt.plot(sdf_['bid_vwap'], label='bid_vwap')
    plt.plot(sdf_['ask_vwap'], label='ask_vwap')
    plt.legend()
    plt.show()
    #%%
    plt.plot(sdf["ask_vol"]-sdf["bid_vol"], label="diff vol")
    plt.plot(sdf["ask_vol"], label="ask vol")
    plt.plot(-sdf["bid_vol"], label="bid vol")
    plt.legend()
    plt.show()
