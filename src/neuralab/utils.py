
from typing import Optional, TypeVar



T = TypeVar('T')
def optional(value: Optional[T], default: T) -> T:
    return default if value is None else value