from pathlib import Path
from typing import Optional

NEURALAB_HOME_PATH = Path.home() / ".neuralab"
# settings
NEURALAB_BUCKET_NAME = "neuralab"
NEURALAB_LOCAL_STORAGE_PATH = Path.home() / ".neuralab"



def storage_path(obj_or_cls: object | type, *parts: str) -> Path:
    cls = type(obj_or_cls) if not isinstance(obj_or_cls, type) else obj_or_cls
    path = NEURALAB_HOME_PATH / cls.__qualname__
    for parh in parts:
        path /= parh
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
