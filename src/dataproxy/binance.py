#%%
from datetime import date as Date
from io import BytesIO
from logging import getLogger as get_logger
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

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

def fetch_agg_trade_dataframe(
    date: Date,
    symbol: str,
) -> pd.DataFrame:

    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date}.zip"

    
    log.info("Downloading {url})")
    with urlopen(url) as url_buffer:
        buffer = url_buffer.read()

    with ZipFile(BytesIO(buffer)) as zip_file:
        with zip_file.open(f"{symbol}-aggTrades-{date}.csv") as csv:
            return pd.read_csv(
                csv, 
                names=_AGG_TRADE_COLUMN_NAMES, 
                dtype=_AGG_TRADE_COLUMN_DTYPE,
                index_col="id"
            )

    return df


#%%

fetch_agg_trade_dataframe(Date.fromisoformat('2021-04-24'), 'ETHUSDT')
