from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from typing import Iterable, Optional

import numpy as np


@dataclass(init=True, repr=True, eq=True)
class Bar:
    """An OHLCV event"""

    open: float = np.nan
    high: float = np.nan
    low: float = np.nan
    close: float = np.nan
    volume: float = np.nan


class BarData:
    """Dictionary like container for a collection of bars at a current time period"""

    def __init__(self, as_of: Optional[datetime] = None) -> None:
        self._bars: dict[str, Bar] = {}
        self._as_of = as_of or datetime.min

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def __getitem__(self, key: str) -> Bar:
        return self._bars[key]

    def __setitem__(self, key: str, bar: Bar) -> None:
        self._bars[key] = bar

    def __repr__(self) -> str:
        return repr(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BarData):
            return False
        return self._bars == value._bars

    def get(self, key: str) -> Bar | None:
        return self._bars.get(key)

    def symbols(self) -> Iterable[str]:
        return self._bars.keys()

    def bars(self) -> Iterable[tuple[str, Bar]]:
        return self._bars.items()


class OrderType(IntEnum):
    """The type of order. Currently on market orders are supported."""

    MARKET = 1


@dataclass(init=True, repr=True, eq=True)
class Order:
    """Information for a new order"""

    id: str
    symbol: str
    quantity: float
    type: OrderType
    submit_time: datetime
    price: Optional[float] = None


@dataclass(init=True, repr=True, eq=True)
class ExecutionReport:
    """Information for an order fill"""

    order: Order
    last_price: float
    last_quantity: float
    cum_quantity: float
    rem_quantity: float
    fill_timestamp: datetime
