# %%
from dataproxy.binance import fetch_sample_dataframe
from timeutils import TimeRange, TimeDeltaParseable
from jax import numpy as np
from pandas import DataFrame

def fetch_dataset(symbols: set[str], years: range, sample_rate: TimeDeltaParseable):

    def time_range_of(year: int, month: int):
        stop_year = year + 1 if month == 12 else year
        stop_month = 1 if month == 12 else month + 1

        return TimeRange.of(
            f"{year}-{month:02}-01", f"{stop_year}-{stop_month:02}-01", sample_rate
        )

    return [
        fetch_sample_dataframe(time_range_of(year, month), symbol)
        for symbol in symbols
        for year in years
        for month in range(1, 13)
    ]

DATASETS = fetch_dataset({"BTCUSDT", "ETHUSDT"}, range(2023, 2024), "1m")