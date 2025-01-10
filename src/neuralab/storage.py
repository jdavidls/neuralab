from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import IO, Any, Literal, Optional, Text, cast
from weakref import WeakValueDictionary

import requests
from google.cloud import storage
from jax import tree
from rich.repr import auto
from rich.text import Text

from neuralab import settings, track, logging

STORAGE_PATH = settings.NEURALAB_HOME_PATH
CLOUD_STORAGE_CLIENT = storage.Client()
CLOUD_STORAGE_BUCKET = CLOUD_STORAGE_CLIENT.bucket(settings.NEURALAB_BUCKET_NAME)

_RESOURCE_LOCK = Lock()
_RESOURCE_FUTURES: WeakValueDictionary[Resource, Future] = WeakValueDictionary()
_RESOURCE_CACHE: WeakValueDictionary[Resource, Any] = WeakValueDictionary()

log = logging.get_logger(__name__)


@dataclass(frozen=True)
class Url:
    url: str

    def download(self, chunk_size=16 * 1024) -> BytesIO:
        content = BytesIO()
        self.download_to(content, chunk_size)
        content.seek(0)
        return content

    def download_to(self, destination: IO[bytes], chunk_size=16 * 1024):
        with requests.get(self.url, stream=True) as resp:
            content_length = resp.headers.get("Content-Length")
            content_length = int(content_length) if content_length is not None else None

            with track.task(
                total=content_length,
                description=f"Downloading {self.url}",
                transient=True,
            ) as task:

                for chunk in resp.iter_content(chunk_size):
                    destination.write(chunk)
                    task.update(advance=len(chunk))


@auto
@dataclass(frozen=True)
class Resource[T](ABC):

    @property
    @abstractmethod
    def type(self) -> type[T]: ...

    @property
    @abstractmethod
    def path(self) -> Path: ...

    @property
    @abstractmethod
    def path_suffix(self) -> str: ...

    @abstractmethod
    def make(self) -> T: ...

    @abstractmethod
    def load(self, buffer: IO[bytes]) -> T: ...

    @abstractmethod
    def dump(self, buffer: IO[bytes], res: T): ...

    @property
    def local_file(self) -> Path:
        return (STORAGE_PATH / self.type.__name__ / self.path).with_suffix(
            self.path_suffix
        )

    @property
    def remote_blob(self) -> storage.Blob:
        return CLOUD_STORAGE_BUCKET.blob(str(self.type.__name__ / self.path))

    def ensure(self, local: bool = True, remote: bool = False, checking: bool = True):
        ensure(self, local=local, remote=remote, checking=checking)

    def fetch(self) -> T:
        return fetch(self)

    def store(self, content: T, local=True, remote: bool = False):
        return store(self, content, local=local, remote=remote)


def ensure[
    T
](
    resource: Resource[T],
    local: bool = True,
    remote: bool = False,
    checking: bool = True,
) -> Optional[T]:
    if not local and not remote:
        return

    local_path_exists = local and resource.local_file.exists()
    remote_blob_exists = remote and resource.remote_blob.exists()

    need_local, need_remote = False, False
    local_content = None
    remote_content = None

    if local:
        if not local_path_exists:
            need_local = True
        elif checking:
            local_content = _check_local(resource)
            if local_content is None:
                need_local = True

    if remote:
        if not remote_blob_exists:
            need_remote = True
        elif checking:
            remote_content = _check_remote(resource)
            if remote_content is None:
                need_remote = True

    if need_local and need_remote:

        content = _make(resource)

        if local:
            _mem_to_local(resource, content)

        if remote:
            _mem_to_remote(resource, content)

        return content

    elif need_local:
        if remote_content is not None:
            _mem_to_local(resource, remote_content)
            return remote_content
        else:
            content = _make(resource)
            _mem_to_local(resource, content)
            return content

    elif need_remote:
        if local_content is not None:
            _mem_to_remote(resource, local_content)
            return local_content
        else:
            content = _make(resource)
            _mem_to_remote(resource, content)
            return content

    return local_content if local_content is not None else remote_content


def ensure_tree(
    resources: Any,
    local: bool = True,
    remote: bool = False,
    checking: bool = True,
    num_workers=settings.DEFAULT_NUM_WORKERS,
) -> Any:
    resources, tree_def = tree.flatten(
        resources, is_leaf=lambda x: isinstance(x, Resource)
    )

    results = track.parallel_map(
        ensure,
        resources,
        local,
        remote,
        checking,
        description="Ensuring resources",
        num_workers=num_workers,
    )

    return tree.unflatten(tree_def, results)


def fetch[
    T
](
    resource: Resource[T],
    ensure_local: bool = True,
    ensure_remote: bool = False,
    checking: bool = False,
) -> T:
    with _RESOURCE_LOCK:
        if resource in _RESOURCE_CACHE:
            return _RESOURCE_CACHE[resource]

        if resource in _RESOURCE_FUTURES:
            fut = _RESOURCE_FUTURES[resource]
            await_fut = True
        else:
            fut = Future()
            _RESOURCE_FUTURES[resource] = fut
            await_fut = False

    if await_fut:
        return fut.result()

    content = None
    if ensure_local or ensure_remote:
        content = ensure(
            resource, local=ensure_local, remote=ensure_remote, checking=checking
        )

    if content is None and resource.local_file.exists():
        content = _local_to_mem(resource)

    if content is None and resource.remote_blob.exists():
        content = _remote_to_mem(resource)

    if content is None:
        content = _make(resource)

    with _RESOURCE_LOCK:
        _RESOURCE_CACHE[resource] = content

    fut.set_result(content)

    return cast(T, content)


def fetch_tree(
    resources: Any,
    num_workers=settings.DEFAULT_NUM_WORKERS,
    description="Fetching resources",
    **kwargs,
) -> Any:

    resources, tree_def = tree.flatten(
        resources, is_leaf=lambda x: isinstance(x, Resource)
    )

    results = track.parallel_map(
        fetch, resources, description=description, num_workers=num_workers, **kwargs
    )

    return tree.unflatten(tree_def, results)


def _check_local[T](resource: Resource[T]) -> Optional[T]:
    try:
        return _local_to_mem(resource)
    except Exception as e:
        log.warning(f"Check failed to load {resource.local_file}")
        resource.local_file.unlink()
        return None


def _check_remote[T](resource: Resource[T]) -> Optional[T]:
    try:
        return _remote_to_mem(resource)
    except Exception as e:
        log.warning(f"Check failed to load {resource.remote_blob}")
        resource.remote_blob.delete()
        return None


def store[
    T
](resource: Resource[T], content: T, local: bool = True, remote: bool = False):
    with _RESOURCE_LOCK:
        _RESOURCE_CACHE[resource] = content

    if local:
        _mem_to_local(resource, content)

    if remote:
        _mem_to_remote(resource, content)


def _make[T](self: Resource[T]) -> T:
    try:
        return self.make()
    except Exception as e:
        e.add_note(f"Failed to make resource {self}")
        raise e


def _remote_to_local[T](self: Resource[T]):
    self.local_file.parent.mkdir(parents=True, exist_ok=True)
    with self.local_file.open("wb") as f:
        _download(self, f)


def _local_to_remote[T](self: Resource[T]):
    with self.local_file.open("rb") as f:
        _upload(self, f, self.local_file.stat().st_size)


def _mem_to_remote[T](self: Resource[T], content: T):
    with BytesIO() as buffer:
        _dump(self, content, buffer)
        buffer.seek(0)
        _upload(self, buffer, len(buffer.getvalue()))


def _mem_to_local[T](self: Resource[T], content: T):
    self.local_file.parent.mkdir(parents=True, exist_ok=True)
    with self.local_file.open("wb") as f:
        _dump(self, content, f)


def _local_to_mem[T](self: Resource[T]) -> T:
    with self.local_file.open("rb") as f:
        return _load(self, f)


def _remote_to_mem[T](self: Resource[T]) -> T:
    with BytesIO() as buff:
        _download(self, buff)
    return self.load(buff)


def _dump[T](self: Resource[T], content: T, destination: IO[bytes]):
    with track.task(
        description=f"Dumping {self.path}",
        total=None,
        transient=True,
    ) as task:
        try:
            self.dump(task.wrap_io("write", destination), content)  # type: ignore

        except Exception as e:
            task.console.log(f"[red]Dump failed {self.path}")
            raise e


def _load[T](self: Resource[T], source: IO[bytes]) -> T:
    with track.task(
        description=f"Loading {self.path}",
        total=self.local_file.stat().st_size,
        transient=True,
    ) as task:
        try:
            return self.load(task.wrap_io("read", source))
        except Exception as e:
            task.console.log(f"[red]Load failed {self.path}")
            raise e


def _upload[T](self: Resource[T], source: IO[bytes], size: int):
    with track.task(
        description=f"Uploading {self.path}",
        total=size,
        transient=True,
    ) as task:
        try:
            self.remote_blob.upload_from_file(task.wrap_io("read", source))
        except Exception as e:
            task.console.log(f"[red]Upload failed {self.path}")
            raise e


def _download[T](self: Resource[T], destination: IO[bytes]):
    blob = self.remote_blob
    blob_size = blob.size

    with track.task(
        description="Downloading {self.path}",
        total=blob_size,
        transient=True,
    ) as task:

        try:
            blob.download_to_file(task.wrap_io("write", destination), raw_download=True)
        except Exception as e:
            task.console.log(f"[red]Download failed {self.path}")
            raise e
