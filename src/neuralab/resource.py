# %%
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import IO, Any, Optional, Self, cast
from weakref import WeakValueDictionary

import requests
from flax import struct
from google.cloud import storage
from jax import tree
from rich.repr import auto

from neuralab import logging, settings, track

STORAGE_PATH = settings.NEURALAB_HOME_PATH
CLOUD_STORAGE_CLIENT = storage.Client()
CLOUD_STORAGE_BUCKET = CLOUD_STORAGE_CLIENT.bucket(settings.NEURALAB_BUCKET_NAME)

_RESOURCE_LOCK = Lock()
_RESOURCE_FUTURES: WeakValueDictionary[Resource.Ref, Future] = WeakValueDictionary()
_RESOURCE_CACHE: WeakValueDictionary[Resource.Ref, Any] = WeakValueDictionary()

log = logging.get_logger(__name__)


@dataclass(frozen=True)
class Url:
    url: str

    def download(self, chunk_size=16 * 1024) -> IO[bytes]:
        payload = BytesIO()
        self.download_to(payload, chunk_size)
        payload.seek(0)
        return BytesIO(payload.getvalue())

    def download_to(self, destination: IO[bytes], chunk_size=16 * 1024):
        with requests.get(self.url, stream=True) as resp:
            resource_length = resp.headers.get("Content-Length")
            resource_length = (
                int(resource_length) if resource_length is not None else None
            )

            with track.task(
                total=resource_length,
                description=f"Downloading {self.url}",
                transient=True,
            ) as task:

                for chunk in resp.iter_content(chunk_size):
                    destination.write(chunk)
                    task.update(advance=len(chunk))


#@auto
#@struct.dataclass
class Resource:  # un dataset es un modulo porque tiene pesos!!

    @auto
    @dataclass(frozen=True)
    class Ref[T](ABC):
        @property
        @abstractmethod
        def type(self) -> type[T]: ...
            
        @property
        @abstractmethod
        def path(self) -> Path: ...

        @abstractmethod
        def make(self) -> T: ...

        @property
        def path_suffix(self):
            return f".pkl"

        def load(self, buffer: IO[bytes]) -> T:
            return pickle.load(buffer)

        def dump(self, buffer: IO[bytes], resource: T):
            pickle.dump(resource, buffer)

        @property
        def local_file(self) -> Path:
            return (STORAGE_PATH / self.type.__name__ / self.path).with_suffix(
                self.path_suffix
            )

        @property
        def remote_blob(self) -> storage.Blob:
            return CLOUD_STORAGE_BUCKET.blob(str(self.type.__name__ / self.path))

        def ensure(
            self, local: bool = True, remote: bool = False, checking: bool = True
        ):
            ensure(self, local=local, remote=remote, checking=checking)

        def fetch(self) -> T:
            return fetch(self)

        def store(self, resource: T, local=True, remote: bool = False):
            return store(self, resource, local=local, remote=remote)

    ref: Resource.Ref[Self] = struct.field(pytree_node=False)

def ensure[
    T
](
    ref: Resource.Ref[T],
    local: bool = True,
    remote: bool = False,
    checking: bool = True,
) -> Optional[T]:
    if not local and not remote:
        return

    local_path_exists = local and ref.local_file.exists()
    remote_blob_exists = remote and ref.remote_blob.exists()

    need_local, need_remote = False, False
    local_resource = None
    remote_resource = None

    if local:
        if not local_path_exists:
            need_local = True
        elif checking:
            local_resource = _check_local(ref)
            if local_resource is None:
                need_local = True

    if remote:
        if not remote_blob_exists:
            need_remote = True
        elif checking:
            remote_resource = _check_remote(ref)
            if remote_resource is None:
                need_remote = True

    if need_local and need_remote:

        resource = _make(ref)

        if local:
            _mem_to_local(ref, resource)

        if remote:
            _mem_to_remote(ref, resource)

        return resource

    elif need_local:
        if remote_resource is not None:
            _mem_to_local(ref, remote_resource)
            return remote_resource
        else:
            resource = _make(ref)
            _mem_to_local(ref, resource)
            return resource

    elif need_remote:
        if local_resource is not None:
            _mem_to_remote(ref, local_resource)
            return local_resource
        else:
            resource = _make(ref)
            _mem_to_remote(ref, resource)
            return resource

    return local_resource if local_resource is not None else remote_resource


def ensure_tree(
    resources: Any,
    local: bool = True,
    remote: bool = False,
    checking: bool = True,
    num_workers=settings.DEFAULT_NUM_WORKERS,
) -> Any:
    resources, tree_def = tree.flatten(
        resources, is_leaf=lambda x: isinstance(x, Resource.Ref)
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
    ref: Resource.Ref[T],
    ensure_local: bool = True,
    ensure_remote: bool = False,
    checking: bool = False,
) -> T:
    with _RESOURCE_LOCK:
        if ref in _RESOURCE_CACHE:
            return _RESOURCE_CACHE[ref]

        if ref in _RESOURCE_FUTURES:
            fut = _RESOURCE_FUTURES[ref]
            await_fut = True
        else:
            fut = Future()
            _RESOURCE_FUTURES[ref] = fut
            await_fut = False

    if await_fut:
        return fut.result()

    resource = None
    if ensure_local or ensure_remote:
        resource = ensure(
            ref, local=ensure_local, remote=ensure_remote, checking=checking
        )

    if resource is None and ref.local_file.exists():
        resource = _local_to_mem(ref)

    if resource is None and ref.remote_blob.exists():
        resource = _remote_to_mem(ref)

    if resource is None:
        resource = _make(ref)

    with _RESOURCE_LOCK:
        _RESOURCE_CACHE[ref] = resource

    fut.set_result(resource)

    return cast(T, resource)


def fetch_tree(
    resources: Any,
    num_workers=settings.DEFAULT_NUM_WORKERS,
    description="Fetching resources",
    **kwargs,
) -> Any:

    resources, tree_def = tree.flatten(
        resources, is_leaf=lambda x: isinstance(x, Resource.Ref)
    )

    results = track.parallel_map(
        fetch, resources, description=description, num_workers=num_workers, **kwargs
    )

    return tree.unflatten(tree_def, results)


def store[
    T
](ref: Resource.Ref[T], resource: T, local: bool = True, remote: bool = False):
    with _RESOURCE_LOCK:
        _RESOURCE_CACHE[ref] = resource

    if local:
        _mem_to_local(ref, resource)

    if remote:
        _mem_to_remote(ref, resource)


def _make[T](self: Resource.Ref[T]) -> T:
    try:
        return self.make()
    except Exception as e:
        e.add_note(f"Failed to make ref {self}")
        raise e


def _check_local[T](ref: Resource.Ref[T]) -> Optional[T]:
    try:
        return _local_to_mem(ref)
    except Exception as e:
        log.warning(f"Check failed to load {ref.local_file}")
        ref.local_file.unlink()
        return None


def _check_remote[T](ref: Resource.Ref[T]) -> Optional[T]:
    try:
        return _remote_to_mem(ref)
    except Exception as e:
        log.warning(f"Check failed to load {ref.remote_blob}")
        ref.remote_blob.delete()
        return None


def _remote_to_local[T](self: Resource.Ref[T]):
    self.local_file.parent.mkdir(parents=True, exist_ok=True)
    with self.local_file.open("wb") as f:
        _download(self, f)


def _local_to_remote[T](self: Resource.Ref[T]):
    with self.local_file.open("rb") as f:
        _upload(self, f, self.local_file.stat().st_size)


def _mem_to_remote[T](self: Resource.Ref[T], resource: T):
    with BytesIO() as buffer:
        _dump(self, resource, buffer)
        buffer.seek(0)
        _upload(self, buffer, len(buffer.getvalue()))


def _mem_to_local[T](self: Resource.Ref[T], resource: T):
    self.local_file.parent.mkdir(parents=True, exist_ok=True)
    with self.local_file.open("wb") as f:
        _dump(self, resource, f)


def _local_to_mem[T](self: Resource.Ref[T]) -> T:
    with self.local_file.open("rb") as f:
        return _load(self, f)


def _remote_to_mem[T](self: Resource.Ref[T]) -> T:
    with BytesIO() as buff:
        _download(self, buff)
    return self.load(buff)


def _dump[T](self: Resource.Ref[T], resource: T, destination: IO[bytes]):
    with track.task(
        description=f"Dumping {self.path}",
        total=None,
        transient=True,
    ) as task:
        try:
            self.dump(task.wrap_io("write", destination), resource)  # type: ignore

        except Exception as e:
            task.console.log(f"[red]Dump failed {self.path}")
            raise e


def _load[T](self: Resource.Ref[T], source: IO[bytes]) -> T:
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


def _upload[T](self: Resource.Ref[T], source: IO[bytes], size: int):
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


def _download[T](self: Resource.Ref[T], destination: IO[bytes]):
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
