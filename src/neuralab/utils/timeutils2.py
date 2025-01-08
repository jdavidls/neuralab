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

# TODAS LAS FECHAS SIEMPRE EN UTC!!!

# Expresiones regulares
TIME_DELTA_REGEX = ...
    # 1w -> 1 semana
    # 1d -> 1 día
    # 1h -> 1 hora
    # 1m -> 1 minuto
    # 1s -> 1 segundo
    # Ejemplo 1h 30m
    # Ejemplo 1w 2d 3h 4m 5s

DATE_TIME_REGEX = ... 
    # TODO Implementar sistema implicito:
    #  2024 -> 2024-01-01T00:00:00
    #  2024-05 -> 2024-05-01T00:00:00
    #  2024-05-10 -> 2024-05-10T00:00:00
    #  2024-05-10T12 -> 2024-05-10T12:00:00
    #  2024-05-10T12:30 -> 2024-05-10T12:30:00
    #  2024-05-10T12:30:15 -> 2024-05-10T12:30:15

TIME_RANGE_REGEX = ...
    # 2024 to 2025 every 1h -> TimeRange(start=2024-01-01, stop=2025-01-01, step=1h)
    # 2024-01-01 to 2024-01-10 every 6h -> TimeRange(start=2024-01-01, stop=2024-01-10, step=6h)
    # 2024 every 1d -> TimeRange(start=2024-01-01, stop=2025-01-01, step=1d)
    # 2024-01 every 1h -> TimeRange(start=2024-01-01, stop=2024-02-01, step=1h)
    # 2024-01-01 every 1m -> TimeRange(start=2024-01-01T00:00:00, stop=2024-01-01T00:01:00, step=1m)


@dataclass
class TimeRange:
    start: DateTime
    stop: DateTime
    step: TimeDelta = TimeDelta(days=1)

    def __str__(self):
        raise NotImplementedError # TODO: implementar la representación de un TimeRange
        # debe detectar si el timerange se puede representar como un rango implicito
        # o explicito y devolverlo en el formato correspondiente
        # Ejemplo: "2024 to 2025 every 1h"
        # Ejemplo: "2024-01-01 to 2024-01-10 every 6h"

if __name__ == '__main__':
    # Ejemplos de uso
    # TODO: implementar ejemplos de uso para parsear fechas, deltas y rangos de tiempo
    # y mostrar el resultado en un formato legible        