from typing import Literal

type Market = Literal["binance-spot", "binance-usdtm", "binance-coinm"]
type Symbol = str


type SymbolSet = frozenset[Symbol]
type MarketSet = frozenset[Market]


SYMBOLSET_TWO_BIGS: SymbolSet = frozenset({
        "ETHUSDT",
        "BTCUSDT",
    })

SYMBOLSET_NAMES: dict[SymbolSet, str] = {
    SYMBOLSET_TWO_BIGS: "two-bigs",
}

MARKETSET_BINANCE: MarketSet = frozenset({
    "binance-spot",
    "binance-usdtm", 
    #"binance-coinm"
})

MARKETSET_NAMES: dict[frozenset[Market], str] = {
    MARKETSET_BINANCE: "binance",
}

DEFAULT_SYMBOLSET: SymbolSet = SYMBOLSET_TWO_BIGS
DEFAULT_MARKETSET: MarketSet = MARKETSET_BINANCE