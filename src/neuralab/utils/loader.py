from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from os import cpu_count
from typing import Callable, Iterator, Optional, cast
from atpbar import atpbar, flushing
from math import ceil
from numpy import str_
import requests

# Iterable with associated length
class IterableWithLength:
    def __init__(self, iter: Iterator, length: int):
        self.iter = iter
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.iter
  

class ResponseWrapper:
    def __init__(self, response: requests.Response, chunk_size: int = 4096):
        self.response = response
        self.chunk_size = chunk_size

    def __len__(self):
        try:
            content_length = cast(str, self.response.headers.get('Content-Length'))
            return ceil(int(content_length) / self.chunk_size)
        except:
            raise TypeError("Content-Length header is missing")

    def __iter__(self):
        return self.response.iter_content(chunk_size=self.chunk_size)
   

class Loader:
    def __init__(self, executor: ThreadPoolExecutor, title: Optional[str] = None):
        self.title=title
        self.thread_pool = executor
        self.futures = {}

    def add_task(self, fn, *args, **kwargs):
        fut = self.thread_pool.submit(fn, *args, **kwargs)
        self.futures[fut] = None

    def download(self, url, title: Optional[str] = None) -> bytes:
        return b"".join(
            data for data in atpbar(
                ResponseWrapper(
                    requests.get(url, stream=True)
                ), 
                name=title
            )
        )
    
    def get_results(self):
        return list(self.futures.values())

    @classmethod
    @contextmanager
    def pool(cls, title: str, max_workers: int | None = cpu_count()):
        with flushing():
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                with Loader(executor, title=title) as downloader:
                    yield downloader

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        completed = IterableWithLength(as_completed(self.futures), len(self.futures))

        for future in atpbar(completed, name=self.title):
            if exc := future.exception():
                raise exc
            self.futures[future] = future.result()        

        return False
    