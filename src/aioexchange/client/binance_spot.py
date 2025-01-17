# %%
"""
Un maracado puede manejar multiples sesiones con el fin de
evitar los rate limits.

https://api.binance.com
https://api-gcp.binance.com
https://api1.binance.com
https://api2.binance.com
https://api3.binance.com
https://api4.binance.com


HTTP Return Codes

HTTP 4XX return codes are used for malformed requests; the issue is on the sender's side.
HTTP 403 return code is used when the WAF Limit (Web Application Firewall) has been violated.
HTTP 409 return code is used when a cancelReplace order partially succeeds. (i.e. if the cancellation of the order fails but the new order placement succeeds.)
HTTP 429 return code is used when breaking a request rate limit.
HTTP 418 return code is used when an IP has been auto-banned for continuing to send requests after receiving 429 codes.
HTTP 5XX return codes are used for internal errors; the issue is on Binance's side. It is important to NOT treat this as a failure operation; the execution status is UNKNOWN and could have been a success.
Previous
General API Information

Error Codes

{
  "code": -1121,
  "msg": "Invalid symbol."
}


"""
from __future__ import annotations

from asyncio import gather
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import wraps
from time import time
from typing import Annotated, Optional, get_type_hints
from inspect import get_annotations
from aiohttp import ClientSession as HttpClientSession, WSMsgType
from pydantic import BaseModel, Field, TypeAdapter, create_model
from torch import Stream
from yarl import URL

from aioexchange.client.binance import model


def get(endpoint: str, *, weight: int = 1):
    def decorator(fn):
        #hints = get_type_hints(fn, include_extras=True)
        hints = get_annotations(fn, eval_str=True)
        print(hints)

        return_type = hints.pop("return", None)
        Args = create_model(f"{fn.__name__}.Args", __module__=fn.__module__, **hints)

        return_adapter = TypeAdapter(return_type)
        argument_adapter = TypeAdapter(Args)

        @wraps(fn)
        async def wrapper(self: BinanceSpotSession, **kwargs):
            params = argument_adapter.validate_python(kwargs)
            params = params.model_dump(by_alias=True)
            params = {k: v for k, v in params.items() if v is not None}
            print(params)

            async with self.http.get(endpoint, params=params) as res:
                res.raise_for_status()
                return return_adapter.validate_json(await res.read())

        wrapper.Args = Args
        return wrapper

    return decorator


class BinanceSpotSession:
    ENDPOINTS = [
        URL("https://api.binance.com"),
        URL("https://api-gcp.binance.com"),
        URL("https://api1.binance.com"),
        URL("https://api2.binance.com"),
        URL("https://api3.binance.com"),
        URL("https://api4.binance.com"),
    ]

    def __init__(self, endpoint: URL):
        self.endpoint = endpoint
        self.http = HttpClientSession(endpoint)
        # self.rate_limits = self.RateLimits()

    @get("/api/v3/ping", weight=1)
    async def _ping(self) -> model.Ping: ...

    async def ping(self, times=10):
        # todo times mean stdev
        start = time()
        await self._ping()
        end = time()
        return end - start

    @get("/api/v3/time", weight=1)
    async def server_time(self) -> model.ServerTime: ...

    @get("/api/v3/aggTrades", weight=2)
    async def aggregate_trades(
        self,
        symbol: Annotated[str, Field()],
        from_id: Annotated[Optional[int], Field(alias="fromId", default=None)],
        start_time: Annotated[Optional[int], Field(alias="startTime", default=None)],
        end_time: Annotated[Optional[int], Field(alias="endTime", default=None)],
        limit: Annotated[Optional[int], Field(default=500)],
    ) -> list[model.AggregateTrade]: ...

class BinanceSpotStreams:
    ENDPOINTS = [
        URL("wss://stream.binance.com:9443"),
        URL("wss://stream.binance.com:443"),
    ]

    def __init__(self, endpoint: URL):
        self.endpoint = endpoint
        self.http = HttpClientSession(endpoint)


    async def connect(self, *streams: str):
        params = {'streams': "/".join(streams)}
        async with self.http.ws_connect(f'/stream', params=params) as ws:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT or WSMsgType.BINARY:
                    event = model.StreamEvent.model_validate_json(msg.data)
                    yield event.payload
                elif msg.type == WSMsgType.ERROR:
                    break

    @asynccontextmanager
    async def connect2(self, *streams: str):
        params = {'streams': "/".join(streams)}
        async with self.http.ws_connect(f'/stream', params=params) as ws:

            async def ws_gen():
                msg = await ws.receive()
                match msg.type:
                    case WSMsgType.TEXT | WSMsgType.BINARY:
                        event = model.StreamEvent.model_validate_json(msg.data)
                        yield event.payload
                    case WSMsgType.ERROR:
                        raise StopAsyncIteration

            yield ws_gen()

# Retrona un diccionario de async generators por cada stream.. y un worker
#

# %%
ses = BinanceSpotSession(URL("https://api.binance.com"))
await ses.aggregate_trades(symbol="BTCUSDT")

#%%
# use aiostream for merge multiple sources (binance futures kraken etc..)
# compose the multi dataframe for each time step
# emit the dataframe to a execution process..
stream_src = BinanceSpotStreams(URL("wss://stream.binance.com:9443"))
async for event in stream_src.connect("btcusdt@aggTrade"):
    print(event)
    ## Diccionarios con conjuntos de eventos, 
    # segun va completando 

def event_handler(sources: set):

    ...




# %%

@dataclass(frozen=True)
class Multi[V]:
    items: dict[tuple, V]

    def __getattr__(self, name):
        return Multi({(k, name): getattr(v, name) for k, v in self.items.items()})

    def __await__(self):
        async def mix():
            futs = [v for v in self.items.values()]
            values = await gather(*futs)
            return Multi({k: v for k, v in zip(self.items.keys(), values)})

        return mix().__await__()

    def __call__(self, *args, **kwargs):
        return Multi({k: v(*args, **kwargs) for k, v in self.items.items()})


multi = Multi({ep: BinanceSpotSession(URL(ep)) for ep in ENDPOINTS})

# ses = BinanceSession(URL("https://api.binance.com"))
# await ses.ping()
