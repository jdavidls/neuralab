from __future__ import annotations
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import time as Time
from datetime import timedelta as TimeDelta
from datetime import timezone as TimeZone
import numpy as np

type TimeDeltaParseable = int | str | TimeDelta
type DateTimeParseable = int | str | Date | DateTime

def millis(time: Date | DateTime | TimeDelta | int) -> int:
    match time:
        case int():
            return time
        case DateTime():
            return int(time.timestamp() * 1000)
        case Date():
            return int(DateTime.combine(time, Time.min, TimeZone.utc).timestamp() * 1000)
        case TimeDelta():
            return int(time.total_seconds() * 1000)
        case _:
            raise ValueError(
                f'Can not calculate milliseconds for type {type(time)}'
            )



_1_DAY = TimeDelta(days=1)
_1_DAY_MS = millis(_1_DAY)

@dataclass
class TimeRange:
    start: DateTime
    stop: DateTime
    step: TimeDelta

    def __len__(self):
        return int((self.stop - self.start) // self.step)

    def __iter__(self):
        start, stop, step = self.start, self.stop, self.step
        while start < stop:
            yield start
            start = start + step

    def __getitem__(self, idx: int):
        return self.start + self.step * idx
    
    def __contains__(self, dt: DateTime):
        return self.start <= dt < self.stop

    def dates(self):
        start, stop = self.start, self.stop
        step = TimeDelta(days=1)

        while start < stop:
            yield start.date()
            start = start + step

    @property
    def extent(self) -> TimeDelta:
        return self.stop - self.start

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(
            millis(self.start),
            millis(self.stop),
            millis(self.step),
            dtype=np.int64
        )
    
    @staticmethod
    def from_date(date: Date, step: TimeDelta):
        start = DateTime.combine(date, Time.min).replace(tzinfo=TimeZone.utc)
        stop = start + _1_DAY
        return TimeRange(start, stop, step)
    
    def with_step(self, step: TimeDelta):
        return TimeRange(self.start, self.stop, step)

    @classmethod
    def of(cls, start: DateTimeParseable, stop: DateTimeParseable, step: TimeDelta):
        return cls(parse_datetime(start), parse_datetime(stop), step)

# @dataclass
# class DateRange:
#     start_at: Date
#     stop_at: Date

#     def range(self, step: TimeDelta):
#         return pt.arange(millis(self.start_at), millis(self.stop_at), millis(step))

#     @property
#     def days(self):
#         return (self.stop_at - self.start_at) // _1_DAY

#     def extent(self, delta: TimeDelta):
#         return (self.stop_at - self.start_at) / delta

#     # def __len__(self):
#     #     return int(self.extent())

#     # def __iter__(self):
#     #     return self.iter()

#     # iteradores:
#     # range

#     def range(self, delta: TimeDelta = _1_DAY, /, last_step=False):
#         assert delta >= _1_DAY
#         a = self.start_at
#         while a < self.stop_at:
#             yield a
#             a = a + delta
#         if last_step:
#             yield a

#     def subrange(self, delta: TimeDelta = _1_DAY, /, last_step=False):
#         assert delta >= _1_DAY
#         a = self.start_at
#         while a < self.stop_at:
#             b = a + delta
#             yield DateRange(a, b)
#             a = b
#         if last_step:
#             yield DateRange(a, a + delta)

#     def step(self, idx: int, unit: TimeDelta = _1_DAY):
#         start_at = self.start_at + unit * idx
#         stop_at = start_at + unit
#         return DateRange(start_at, stop_at)

#     def to_timerange(self) -> TimeRange:
#         return TimeRange(
#             DateTime.combine(self.start_at, Time.min),
#             DateTime.combine(self.stop_at, Time.min),
#         )

#     @staticmethod
#     def from_str(start_at: str, stop_at: str):
#         return DateRange(
#             Date.fromisoformat(start_at),
#             Date.fromisoformat(stop_at)
#         )


# @dataclass
# class TimeRange:
#     start_at: DateTime
#     stop_at: DateTime

#     def __len__(self):
#         return self.extent

#     @property
#     def extent(self):
#         return self.stop_at - self.start_at

#     @property
#     def utc(self):
#         return self.tz(TimeZone.utc)

#     def tz(self, tz: TimeZone):
#         return TimeRange(
#             self.start_at.replace(tzinfo=tz),
#             self.stop_at.replace(tzinfo=tz),
#         )

#     def iter_step(self, delta: TimeDelta = _1_DAY, /, last_step=False):
#         '''
#             Arroja dateranges (from & stop) por cada time delta comprendido
#             entre start y stop
#         '''
#         # if delta is None:
#         # delta = TimeDelta(**kwargs)

#         a = self.start_at
#         while a < self.stop_at:
#             b = a + delta
#             yield TimeRange(a, b)
#             a = b

#         if last_step:
#             yield TimeRange(a, a + delta)

#     def step(self, unit: TimeDelta, idx: int):
#         start_at = self.start_at + unit * idx
#         stop_at = start_at + unit
#         return TimeRange(start_at, stop_at)

#     def __str__(self):
#         return f'[{self.start_at.isoformat()},{self.stop_at.isoformat()})'

#     def __index__(self):
#         return slice(self.start_at, self.stop_at)

#     def __contains__(self):
#         ...

#     def dump(self):
#         return {
#             'start_at': self.start_at.isoformat(),
#             'stop_at': self.stop_at.isoformat(),
#         }

#     @classmethod
#     def parse(cls, start_at: str, stop_at: str, **_):
#         return TimeRange(
#             DateTime.fromisoformat(start_at),
#             DateTime.fromisoformat(stop_at),
#         )

#     @classmethod
#     def from_iso_format(
#         cls,
#         start_at: int | str | DateTime,
#         stop_at: int | str | DateTime,
#     ):
#         return cls(_ensure_date_time(start_at), _ensure_date_time(stop_at))


def parse_datetime(dt: DateTimeParseable) -> DateTime:
    if isinstance(dt, int):
        return DateTime.fromtimestamp(dt).replace(tzinfo=TimeZone.utc)
    elif isinstance(dt, str):
        try:
            return DateTime.fromisoformat(dt).replace(tzinfo=TimeZone.utc)
        except:
            dt = Date.fromisoformat(dt)
    
    if isinstance(dt, Date):
        return DateTime.combine(dt, Time.min).replace(tzinfo=TimeZone.utc)
    elif isinstance(dt, DateTime):
        return DateTime.fromtimestamp(dt.timestamp()).replace(tzinfo=TimeZone.utc)
    else:
        raise ValueError(f'Invalid type {type(dt)}')

import re

_TIMEDELTA_REGEX = re.compile(
    r'^'
    r'((?P<days>\d+)d)?'
    r'((?P<hours>\d+)h)?'
    r'((?P<minutes>\d+)m)?'
    r'((?P<seconds>\d+)s)?'
    r'$'
)



def parse_timedelta(s: TimeDeltaParseable) -> TimeDelta:
    if isinstance(s, TimeDelta):
        return s
    if isinstance(s, int):
        return TimeDelta(milliseconds=s)
    
    match = _TIMEDELTA_REGEX.match(s)
    if not match:
        raise ValueError(f'Invalid timedelta string: {s}')
    
    kwargs = {k: int(v) for k, v in match.groupdict().items() if v is not None}
    return TimeDelta(**kwargs)


def _dt_to_ts(dt: DateTime):
    'DateTime to timestamp (milliseconds)'
    return int(dt.timestamp()*1000)


def _ts_to_dt(ts: int):
    'timestamp (milliseconds) to DateTime'
    return DateTime.fromtimestamp(ts/1000)


def _td_to_ms(dt: TimeDelta):
    'timedelta to milliseconds'
    return int(dt.total_seconds()*1000)


def _ms_to_td(ms: int):
    'milliseconds to timedelta'
    return TimeDelta(milliseconds=ms)

