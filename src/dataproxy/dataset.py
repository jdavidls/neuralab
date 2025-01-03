# %%
from math import log
from dataproxy.binance import fetch_sample_dataframe
from timeutils import TimeRange, TimeDeltaParseable
from jax import numpy as jnp
from pandas import DataFrame
from flax import struct

def fetch_dataset(symbols: set[str], years: range, sample_rate: TimeDeltaParseable):

    def time_range_of(year: int, month: int):
        stop_year = year + 1 if month == 12 else year
        stop_month = 1 if month == 12 else month + 1

        return TimeRange.of(
            f"{year}-{month:02}-01", f"{stop_year}-{stop_month:02}-01", sample_rate
        )
        
    def prepare_dataset(sample_dataframe):
        log_high_price = jnp.log(sample_dataframe["high"].to_numpy())
        log_low_price = jnp.log(sample_dataframe["low"].to_numpy())
        log_price = jnp.log(sample_dataframe["vwap"].to_numpy())
        log_volume = jnp.log(sample_dataframe["vol"].to_numpy())
        log_ask_volume = jnp.log(sample_dataframe["ask_vol"].to_numpy())
        log_bid_volume = jnp.log(sample_dataframe["bid_vol"].to_numpy())

        log_volume_imbalance = log_ask_volume - log_bid_volume

        diff_log_price = jnp.diff(log_price, prepend=log_price[0])

        return Dataset(
            #log_high_price=log_high_price,
            #log_low_price=log_low_price,
            log_price=log_price,
            diff_log_price=diff_log_price,
            log_volume=log_volume,
            #log_ask_volume=log_ask_volume,
            #log_bid_volume=log_bid_volume,
            log_volume_imbalance=log_volume_imbalance,
            #log_price_diff=log_price
        )

    return [
        prepare_dataset(fetch_sample_dataframe(time_range_of(year, month), symbol))
        for symbol in symbols
        for year in years
        for month in range(1, 13)
    ]

class Dataset(struct.PyTreeNode):
    #log_high_price: jnp.ndarray
    #log_low_price: jnp.ndarray
    log_price: jnp.ndarray
    diff_log_price: jnp.ndarray
    log_volume: jnp.ndarray
    #log_ask_volume: jnp.ndarray
    #log_bid_volume: jnp.ndarray
    log_volume_imbalance: jnp.ndarray

    def __len__(self):
        return len(self.log_price)

    def __getitem__(self, idx):
        return Dataset(
            #log_high_price=self.log_high_price[idx],
            #log_low_price=self.log_low_price[idx],
            log_price=self.log_price[idx],
            diff_log_price=self.diff_log_price[idx],
            log_volume=self.log_volume[idx],
            #log_ask_volume=self.log_ask_volume[idx],
            #log_bid_volume=self.log_bid_volume[idx],
            log_volume_imbalance=self.log_volume_imbalance[idx],
        )

    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(self.log_price)
        plt.show()

DATASETS = fetch_dataset({"BTCUSDT", "ETHUSDT"}, range(2023, 2024), "1m")