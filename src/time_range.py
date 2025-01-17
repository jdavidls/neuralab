# %%
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import time as Time
from datetime import timedelta as TimeDelta
from datetime import timezone as TimeZone
from datetime import UTC
from typing import Optional

import numpy as np
import re


type TimeDeltaParseable = int | str | TimeDelta
type DateTimeParseable = int | str | Date | DateTime


def milliseconds(time: Date | DateTime | TimeDelta | int) -> int:
    match time:
        case int():
            return time
        case DateTime():
            return int(time.timestamp() * 1000)
        case Date():
            return int(
                DateTime.combine(time, Time.min, UTC).timestamp() * 1000
            )
        case TimeDelta():
            return int(time.total_seconds() * 1000)
        case _:
            raise ValueError(f"Can not calculate milliseconds for type {type(time)}")


_1_DAY = TimeDelta(days=1)
_1_DAY_MS = milliseconds(_1_DAY)


@dataclass(frozen=True, slots=True)
class TimeRange:
    start: DateTime
    stop: DateTime
    step: TimeDelta = _1_DAY

    type Parseable = str | TimeRange


    @property
    def is_for_an_exact_day(self) -> Optional[Date]:
        if datetime_is_day_exact(self.start) and self.stop == datetime_next_day(self.start):
            return self.start.date()

    @property
    def is_for_exact_days(self) -> bool:
        return datetime_is_day_exact(self.start) and datetime_is_day_exact(self.stop)

    @property
    def is_for_an_exact_month(self) -> bool:
        return datetime_is_month_exact(self.start) and self.stop == datetime_next_month(
            self.start
        )

    @property
    def is_for_exact_months(self) -> bool:
        return datetime_is_month_exact(self.start) and datetime_is_month_exact(self.stop)

    @property
    def is_for_an_exact_year(self) -> bool:
        return datetime_is_year_exact(self.start) and self.stop == datetime_next_year(
            self.start
        )

    @property
    def is_for_exact_years(self) -> bool:
        return datetime_is_year_exact(self.start) and datetime_is_year_exact(self.stop)

    @property
    def extent(self) -> TimeDelta:
        return self.stop - self.start

    @property
    def ms(self):
        return range(
            milliseconds(self.start),
            milliseconds(self.stop),
            milliseconds(self.step),
        )

    @staticmethod
    def from_date(date: Date, step: TimeDeltaParseable):
        start = DateTime.combine(date, Time.min).replace(tzinfo=TimeZone.utc)
        stop = start + _1_DAY
        return TimeRange(start, stop, parse_timedelta(step))

    @classmethod
    def combine(
        cls, start: DateTimeParseable, stop: DateTimeParseable, step: TimeDeltaParseable
    ):
        return cls(parse_datetime(start), parse_datetime(stop), parse_timedelta(step))

    @classmethod
    def of(cls, s: Parseable):
        if isinstance(s, TimeRange):
            return s
        if isinstance(s, str):
            return parse_timerange(s)

        raise ValueError(f"Invalid TimeRange string: {s}")

    def __str__(self):
        if self.is_for_an_exact_year:
            prefix = f"{self.start.year}"
        elif self.is_for_an_exact_month:
            prefix = f"{self.start.year}-{self.start.month:02}"
        elif self.is_for_an_exact_day:
            prefix = f"{self.start.date()}"
        else:
            prefix = f"{repr_datetime(self.start)} to {repr_datetime(self.stop)}"

        if self.step == _1_DAY:
            return prefix
        return f"{prefix} {repr_timedelta(self.step)}"

    @property
    def real_length(self) -> float:
        return (self.stop - self.start) / self.step

    def __len__(self):
        return int((self.stop - self.start) // self.step)

    def __iter__(self):
        start, stop, step = self.start, self.stop, self.step

        if step > TimeDelta():
            while start < stop:
                yield start
                start = start + step

        elif step < TimeDelta():
            while stop > start:
                yield stop
                stop += step

        else:
            raise ValueError("TimeRange step must be non-zero step")

    def each(self, interval: TimeDelta):
        start, stop = self.start, self.stop

        if interval > TimeDelta():
            a = start
            b = start + interval
            while a < stop:
                yield self.replace(start=a, stop=b)
                a, b = b, b + interval

        elif interval < TimeDelta():
            a = stop + interval
            b = stop
            while b > start:
                yield self.replace(start=a, stop=b)
                a, b = a + interval, a

        else:
            raise ValueError("TimeRange step must be non-zero step")

    def __getitem__(self, idx: int | slice):

        if isinstance(idx, slice):
            start = self.start + self.step * idx.start if idx.start else self.start
            stop = self.start + self.step * idx.stop if idx.stop else self.stop
            step = self.step * idx.step if idx.step else self.step
            return TimeRange(start, stop, step)

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        return self.start + self.step * idx

    def __contains__(self, dt: DateTime):
        return self.start <= dt < self.stop
    
    def replace(self, **kwargs):
        return replace(self, **kwargs)



# Expresiones regulares
DELTA_REGEX = re.compile(
    r"^(?:(?P<days>\d+)d)?\s?"  # Días opcionales
    r"(?:(?P<hours>\d+)h)?\s?"  # Horas opcionales
    r"(?:(?P<minutes>\d+)m)?\s?"  # Minutos opcionales
    r"(?:(?P<seconds>\d+)s)?$\s?"  # Segundos opcionales
)


def parse_timedelta(s: TimeDeltaParseable) -> TimeDelta:
    if isinstance(s, TimeDelta):
        return s
    if isinstance(s, int):
        return TimeDelta(milliseconds=s)

    match = DELTA_REGEX.match(s)
    if not match:
        raise ValueError(f"Invalid timedelta string: {s}")

    return TimeDelta(
        **{key: int(value) for key, value in match.groupdict(default="0").items()}
    )


DATE_REGEXES = [
    re.compile(r"^(?P<year>\d{4})$"),  # Solo año
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$"),  # Año y mes
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$"),  # Año, mes y día
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$"),  # Fecha y hora
]


def parse_datetime(dt: DateTimeParseable) -> DateTime:
    if isinstance(dt, int):
        return DateTime.fromtimestamp(dt).replace(tzinfo=TimeZone.utc)
    elif isinstance(dt, str):
        try:
            for regex in DATE_REGEXES:
                if match := regex.match(dt):
                    components = match.groupdict()
                    return DateTime(
                        year=int(components["year"]),
                        month=int(components.get("month", 1)),
                        day=int(components.get("day", 1)),
                        hour=int(components.get("hour", 0)),
                        minute=int(components.get("minute", 0)),
                        second=int(components.get("second", 0)),
                        tzinfo=UTC,
                    )
            raise ValueError(f"Invalid date time string: {dt}")

        except:
            dt = Date.fromisoformat(dt)

    if isinstance(dt, Date):
        return DateTime.combine(dt, Time.min).replace(tzinfo=UTC)
    elif isinstance(dt, DateTime):
        return DateTime.fromtimestamp(dt.timestamp()).replace(tzinfo=UTC)
    else:
        raise ValueError(f"Invalid type {type(dt)}")


RANGE_REGEX = re.compile(
    r"^"
    # Rango explícito "start to stop"
    r"((?P<start>.+?)\s+to\s+(?P<stop>.+?)(\s)?)"
    r"$"
)

RANGE_SEP = re.compile(r"[\s_]+")


def parse_timerange(input_str: str) -> TimeRange:
    """
    Parsea una cadena de entrada para determinar el rango de tiempo y el delta.
    """
    input_str = input_str.strip()

    *range_parts, delta_part = RANGE_SEP.split(input_str)
    step = parse_timedelta(delta_part)

    input_str = " ".join(range_parts)
    # Detectar rango explícito
    range_match = RANGE_REGEX.match(input_str)
    if range_match:
        start = parse_datetime(range_match.group("start"))
        stop = parse_datetime(range_match.group("stop"))
        return TimeRange(start=start, stop=stop, step=step)

    # Detectar rango implícito (un solo valor)
    start = parse_datetime(input_str)
    if len(input_str) == 4:  # Año
        stop = datetime_next_year(start)
    elif len(input_str) == 7:  # Año y mes
        stop = datetime_next_month(start)
    elif len(input_str) == 10:  # Año, mes y día
        stop = datetime_next_day(start)
    else:
        raise ValueError(f"Invalid TimeRange string: {input_str}")

    return TimeRange(start=start, stop=stop, step=step)


def repr_timedelta(time_delta: TimeDelta):
    parts = []
    if time_delta < TimeDelta():
        time_delta = -time_delta
        parts.append("-")

    weeks, days = divmod(time_delta.days, 7)
    hours, seconds = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if weeks:
        parts.append(f"{weeks}w")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")

    return "".join(parts)


def repr_datetime(dt: DateTime) -> str:
    if datetime_is_year_exact(dt):
        return str(dt.year)
    if datetime_is_month_exact(dt):
        return f"{dt.year}-{dt.month:02}"
    if datetime_is_day_exact(dt):
        return str(dt.date())
    return dt.isoformat()


def datetime_is_day_exact(dt: DateTime) -> bool:
    return dt.time() == Time.min


def datetime_is_month_exact(dt: DateTime) -> bool:
    return dt.day == 1 and datetime_is_day_exact(dt)


def datetime_is_year_exact(dt: DateTime) -> bool:
    return dt.month == 1 and datetime_is_month_exact(dt)


def datetime_next_day(dt: DateTime) -> DateTime:
    return dt + _1_DAY


def datetime_next_month(dt: DateTime) -> DateTime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    return dt.replace(month=dt.month + 1)


def datetime_next_year(dt: DateTime) -> DateTime:
    return dt.replace(year=dt.year + 1)


# Test cases
if __name__ == "__main__":
    assert parse_timedelta("1d") == TimeDelta(days=1)
    assert parse_timedelta("1d 2h 3m 4s") == TimeDelta(days=1, hours=2, minutes=3, seconds=4)
    assert parse_timedelta("1d 2h 3m") == TimeDelta(days=1, hours=2, minutes=3)

    assert TimeRange.of('2024 1d').is_for_an_exact_year


    examples = [
        "2024 1m",  # Año completo, paso predeterminado (1 día)
        "2024-01 5s",  # Mes completo, paso predeterminado (1 día)
        "2024-01-01",  # Día completo, paso predeterminado (1 día)
        "2024-01-01T12:00",  # Fecha y hora exacta
        "2024-01 to 2024-07 1h",  # Rango de un año, paso de 1 hora
        "2024-01-01 to 2024-01-10 6h",  # Rango explícito con delta
    ]
    
    

    for example in examples:
        try:
            result = TimeRange.of(example)
            print(f"Input: {example}\nParsed: {result}\n")
        except ValueError as e:
            print(f"Input: {example}\nError: {e}\n")

# %%
