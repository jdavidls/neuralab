from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BufferedReader, BufferedWriter
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from weakref import WeakValueDictionary

from google.cloud import storage
from jax import tree
from tqdm import tqdm

from neuralab import settings

STORAGE_PATH = settings.NEURALAB_HOME_PATH
CLOUD_STORAGE_CLIENT = storage.Client()
CLOUD_STORAGE_BUCKET = CLOUD_STORAGE_CLIENT.bucket(settings.NEURALAB_BUCKET_NAME)


def upload_blob(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    blob = CLOUD_STORAGE_BUCKET.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    blob = CLOUD_STORAGE_BUCKET.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


_RESOURCE_FUTURES_LOCK = Lock()
_RESOURCE_FUTURES: WeakValueDictionary[Resource, Future] = WeakValueDictionary()
_RESOURCE_CACHE: WeakValueDictionary[Resource, Any] = WeakValueDictionary()


@dataclass
class Resource[T](ABC):
    @property
    @abstractmethod
    def type(self) -> type[T]: ...

    @property
    @abstractmethod
    def path(self) -> Path: ...

    @abstractmethod
    def make(self) -> T: ...

    @abstractmethod
    def load(self, buffer: BufferedReader) -> T: ...

    @abstractmethod
    def dump(self, buffer: BufferedWriter, res: T): ...

    @property
    def local_path(self) -> Path:
        return STORAGE_PATH / self.type.__name__ / self.path

    @property
    def blob(self) -> storage.Blob:
        return CLOUD_STORAGE_BUCKET.blob(str(self.type.__name__ / self.path))

    def ensure_local(self) -> Optional[T]:
        _ensure_local(self)

    def fetch(self) -> T:
        if self in _RESOURCE_CACHE:
            return _RESOURCE_CACHE[self]

        fut = Future()
        await_fut = False

        with _RESOURCE_FUTURES_LOCK:
            if self in _RESOURCE_FUTURES:
                fut = _RESOURCE_FUTURES[self]
                await_fut = True
            else:
                _RESOURCE_FUTURES[self] = fut

        if await_fut:
            return fut.result()

        content = _fetch_resource(self)

        fut.set_result(content)
        return content

    @staticmethod
    def fetch_tree(resources: Any, max_workers=16):

        resources, tree_def = tree.flatten(
            resources, is_leaf=lambda x: isinstance(x, Resource)
        )

        results = [None] * len(resources)

        if len(resources) > 1:
            with tqdm(total=len(resources), desc="Fetching resources") as progbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:

                    def fetch_task(position, resource: Resource):
                        return position, resource.fetch()

                    ## ATOMIC FETCH
                    def submit_task(pos, res):
                        if res in _RESOURCE_FUTURES:
                            return _RESOURCE_FUTURES[res]

                        fut = executor.submit(fetch_task, pos, res)
                        _RESOURCE_FUTURES[res] = fut
                        return fut

                    with _RESOURCE_FUTURES_LOCK:
                        futures = [
                            submit_task(pos, res) for pos, res in enumerate(resources)
                        ]

                    for future in as_completed(futures):
                        pos, content = future.result()
                        results[pos] = content
                        progbar.update(1)

        return tree.unflatten(tree_def, results)


def _ensure_local[T](self: Resource[T]) -> Optional[T]:
    if self.local_path.exists():
        return

    if self.blob.exists():
        blob = self.blob
        blob_size = blob.size
        with self.local_path.open("wb") as f:
            with tqdm.wrapattr(
                f,
                "write",
                total=blob_size,
                desc=f"Downloading {self.path}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=True,
            ):
                blob.download_to_file(f, raw_download=True)

    content: T = self.make()
    with self.local_path.open("wb") as f:
        self.dump(f, content)

    _RESOURCE_CACHE[self] = content

    return content


def _fetch_resource[T](self: Resource[T]) -> T:
    if self in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[self]

    content = self.ensure_local()
    if content is not None:
        return content

    with open(self.local_path, "rb") as f:
        content = self.load(f)

    _RESOURCE_CACHE[self] = content
    return content
