# %%
from __future__ import annotations
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime as DateTime
from datetime import time as Time
from datetime import timedelta as TimeDelta
from datetime import timezone as TimeZone
import numpy as np
import re


type TimeDeltaParseable = int | str | TimeDelta
type DateTimeParseable = int | str | Date | DateTime
type TimeRangeParseable = str | TimeRange


def milliseconds(time: Date | DateTime | TimeDelta | int) -> int:
    match time:
        case int():
            return time
        case DateTime():
            return int(time.timestamp() * 1000)
        case Date():
            return int(
                DateTime.combine(time, Time.min, TimeZone.utc).timestamp() * 1000
            )
        case TimeDelta():
            return int(time.total_seconds() * 1000)
        case _:
            raise ValueError(f"Can not calculate milliseconds for type {type(time)}")


_1_DAY = TimeDelta(days=1)
_1_DAY_MS = milliseconds(_1_DAY)


def format_timedelta(time_delta: TimeDelta):

    weeks, days = divmod(time_delta.days, 7)
    hours, seconds = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
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


@dataclass
class TimeRange:
    start: DateTime
    stop: DateTime
    step: TimeDelta = _1_DAY

    def __str__(self):
        return (
            f"since {self.start.isoformat()}"
            f" to {self.stop.isoformat()}"
            f" every {format_timedelta(self.step)}"
        )

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
            milliseconds(self.start), milliseconds(self.stop), milliseconds(self.step), dtype=np.int64
        )

    @staticmethod
    def from_date(date: Date, step: TimeDeltaParseable):
        start = DateTime.combine(date, Time.min).replace(tzinfo=TimeZone.utc)
        stop = start + _1_DAY
        return TimeRange(start, stop, parse_time_delta(step))

    def with_step(self, step: TimeDelta):
        return TimeRange(self.start, self.stop, step)

    @classmethod
    def combine(
        cls, start: DateTimeParseable, stop: DateTimeParseable, step: TimeDeltaParseable
    ):
        return cls(parse_date_time(start), parse_date_time(stop), parse_time_delta(step))

    @classmethod
    def of(cls, s: TimeRangeParseable):
        if isinstance(s, TimeRange):
            return s
        if isinstance(s, str):
            return parse_time_range(s)

        raise ValueError(f"Invalid TimeRange string: {s}")



# Expresiones regulares
DELTA_REGEX = re.compile(
    r"^(?:(?P<days>\d+)d)?"  # Días opcionales
    r"(?:(?P<hours>\d+)h)?"  # Horas opcionales
    r"(?:(?P<minutes>\d+)m)?"  # Minutos opcionales
    r"(?:(?P<seconds>\d+)s)?$"  # Segundos opcionales
)

RANGE_REGEX = re.compile(
    r"^(?P<start>.+?)\s+to\s+(?P<stop>.+?)$"  # Rango explícito "start to stop"
)


DATE_REGEXES = [
    re.compile(r"^(?P<year>\d{4})$"),  # Solo año
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})$"),  # Año y mes
    re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$"),  # Año, mes y día
    re.compile(
        r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$"
    ),  # Fecha y hora
]

def parse_time_delta(s: TimeDeltaParseable) -> TimeDelta:
    if isinstance(s, TimeDelta):
        return s
    if isinstance(s, int):
        return TimeDelta(milliseconds=s)

    match = DELTA_REGEX.match(s)
    if not match:
        raise ValueError(f"Invalid timedelta string: {s}")

    return TimeDelta(
        **{
            key: int(value)
            for key, value in match.groupdict(default="0").items()
        }
    )


def parse_date_time(dt: DateTimeParseable) -> DateTime:
    if isinstance(dt, int):
        return DateTime.fromtimestamp(dt).replace(tzinfo=TimeZone.utc)
    elif isinstance(dt, str):
        try:
            for regex in DATE_REGEXES:
                if match := regex.match(dt):
                    components = {
                        key: int(value) for key, value in match.groupdict(default="1").items()
                    }
                    return DateTime(
                        year=components["year"],
                        month=components.get("month", 1),
                        day=components.get("day", 1),
                        hour=components.get("hour", 0),
                        minute=components.get("minute", 0),
                        second=components.get("second", 0),
                    ).replace(tzinfo=TimeZone.utc)
            raise ValueError(f"Invalid date time string: {dt}")
                
        except:
            dt = Date.fromisoformat(dt)

    if isinstance(dt, Date):
        return DateTime.combine(dt, Time.min).replace(tzinfo=TimeZone.utc)
    elif isinstance(dt, DateTime):
        return DateTime.fromtimestamp(dt.timestamp()).replace(tzinfo=TimeZone.utc)
    else:
        raise ValueError(f"Invalid type {type(dt)}")


def parse_time_range(input_str: str) -> TimeRange:
    """
    Parsea una cadena de entrada para determinar el rango de tiempo y el delta.
    """
    delta = TimeDelta(days=1)  # Valor por defecto

    # Separar el delta si está presente al final
    if " " in input_str:
        *range_parts, delta_part = input_str.split(" ")
        input_str = " ".join(range_parts)
        delta = parse_time_delta(delta_part)

    # Detectar rango explícito
    range_match = RANGE_REGEX.match(input_str)
    if range_match:
        start = parse_date_time(range_match.group("start"))
        stop = parse_date_time(range_match.group("stop"))
        return TimeRange(start=start, stop=stop, step=delta)

    # Detectar rango implícito (un solo valor)
    start = parse_date_time(input_str)
    if len(input_str) == 4:  # Solo año
        stop = DateTime(start.year + 1, 1, 1).replace(tzinfo=TimeZone.utc)
    elif len(input_str) == 7:  # Año y mes
        next_month = start.month % 12 + 1
        next_year = start.year + (1 if next_month == 1 else 0)
        stop = DateTime(next_year, next_month, 1).replace(tzinfo=TimeZone.utc)
    else:  # Fecha completa
        stop = start + TimeDelta(days=1)

    return TimeRange(start=start, stop=stop, step=delta)


def represent_time_delta(delta: TimeDelta) -> str:
    """
    Representa un objeto timedelta como un string en el formato definido.
    Ejemplo: "1d 6h 30m".
    """
    days = delta.days
    seconds = delta.seconds
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0:
        parts.append(f"{seconds}s")

    return " ".join(parts)



# Ejemplos de uso
if __name__ == "__main__":
    examples = [
        "2024",  # Año completo, paso predeterminado (1 día)
        "2024-01",  # Mes completo, paso predeterminado (1 día)
        "2024-01-01",  # Día completo, paso predeterminado (1 día)
        "2024-01-01T12:00",  # Fecha y hora exacta
        "2024 to 2025 1h",  # Rango de un año, paso de 1 hora
        "2024-01-01 to 2024-01-10 6h",  # Rango explícito con delta
    ]

    for example in examples:
        try:
            result = TimeRange.parse(example)
            print(f"Input: {example}\nParsed: {result}\n")
        except ValueError as e:
            print(f"Input: {example}\nError: {e}\n")
