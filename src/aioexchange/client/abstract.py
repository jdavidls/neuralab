#%%

from abc import ABC
from contextlib import asynccontextmanager
from typing import ClassVar
from aiohttp import ClientSession as HttpClientSession


class ClientSession(HttpClientSession, ABC):
    client_id: ClassVar[str]

    class RateLimits(ABC):
        ...
        
    def __init__(self):
        self.__http = HttpClientSession()
        self.__rate_limits = self.RateLimits()

    @asynccontextmanager
    async def request_weights(self, weights: dict[str, int]):
        yield self.__http
