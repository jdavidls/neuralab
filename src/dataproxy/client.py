
from datetime import datetime as DateTime

class TimeRange:
    start: DateTime
    stop: DateTime

class DataProxyClient:
    def __init__(self):
        ...

    def fetch_trade_dataset(self, time_range: TimeRange): 
        ...

    