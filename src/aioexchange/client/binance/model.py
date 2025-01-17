from decimal import Decimal
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# name: Annotated[str, Field(strict=True), WithJsonSchema({'extra': 'data'})]


class Ping(BaseModel):
    pass


class ServerTime(BaseModel):
    server_time: Annotated[int, Field(alias="serverTime")]


class AggregateTrade(BaseModel):
    id: Annotated[int, Field(alias="a")]
    price: Annotated[Decimal, Field(alias="p")]
    quantity: Annotated[Decimal, Field(alias="q")]
    first_trade_id: Annotated[int, Field(alias="f")]
    last_trade_id: Annotated[int, Field(alias="l")]
    time: Annotated[int, Field(alias="T")]
    is_bid: Annotated[bool, Field(alias="m")]
    is_best_match: Annotated[bool, Field(alias="M")]


class AggregteTradeEvent(AggregateTrade):
    event: Annotated[Literal["aggTrade"], Field(alias="e")]
    event_time: Annotated[int, Field(alias="E")]
    symbol: Annotated[str, Field(alias="s")]


class BookTickerEvent(BaseModel):
    #event: Annotated[Literal["bookTicker"], Field(alias="e")]
    book_update_id: Annotated[int, Field(alias="u")]
    symbol: Annotated[str, Field(alias="s")]
    best_bid_price: Annotated[Decimal, Field(alias="b")]
    best_bid_quantity: Annotated[Decimal, Field(alias="B")]
    best_ask_price: Annotated[Decimal, Field(alias="a")]
    best_ask_quantity: Annotated[Decimal, Field(alias="A")]


class StreamEvent(BaseModel):
    stream: Annotated[str, Field()]
    payload: Annotated[AggregteTradeEvent, Field(discriminator="event", alias="data")]
