from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, override
from zoneinfo import ZoneInfo

from pathlib import Path
import os
import pandas as pd

from systrade.data import Bar, BarData
from systrade.history import HistoryProvider

import alpaca.data as ad
import pandas as pd
from systrade.data import Bar, BarData
import logging
import time

logger = logging.getLogger(__name__)

class Feed(ABC):
    @abstractmethod
    def start(self) -> None:
        """Start streaming"""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming"""

    @abstractmethod
    def is_running(self) -> bool:
        """Whether feed is currently running"""

    @abstractmethod
    def subscribe(self, symbol: str) -> None:
        """Subscribe to a symbol"""

    @abstractmethod
    def next_data(self) -> BarData:
        """Block until returning the next available data for subscribed
        symbols"""

class FileFeed(Feed):
    def __init__(
        self, path: str | Path, start: Optional[str] = None, end: Optional[str] = None
    ) -> None:
        """File feed initializer

        Parameters
        ----------
        path
            Full path to data file
        start, optional
            When to start the replay, in YYYY-MM-DD format
        end, optional
            When to end the replay, in YYYY-MM-DD format
        """
        self._path = path
        self._data = pd.DataFrame()
        self._available_symbols = set()
        self._subscribed_symbols = BarData()
        self._timestamp_iter = None
        self._current_ts = pd.Timestamp(year=1970, month=1, day=1)
        self._is_running = False
        self._start = start
        self._end = end

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    @override
    def start(self) -> None:
        df = pd.read_csv(self._path)
        df["Date"] = df["Date"].astype(pd.DatetimeTZDtype(tz="America/New_York"))
        df["Symbol"] = df["Symbol"].astype(pd.StringDtype(storage="python"))
        if self._start is not None:
            df = df.loc[df["Date"] >= self._start]
        if self._end is not None:
            df = df.loc[df["Date"] <= self._end]

        self._data = df.set_index(["Symbol", "Date"]).sort_index()
        self._available_symbols = set(self._data.index.get_level_values(0).unique())
        self._timestamp_iter = iter(self._data.index.get_level_values(1).unique())
        self._is_running = True
        try:
            self._current_ts = next(self._timestamp_iter)
            self._is_running = True
        except StopIteration:
            self._is_running = False

    @override
    def stop(self) -> None:
        self._is_running = False

    @override
    def is_running(self) -> bool:
        return self._is_running

    @override
    def subscribe(self, symbol: str) -> None:
        if symbol not in self._available_symbols:
            raise ValueError(f"Symbol {symbol} not available!")
        self._subscribed_symbols[symbol] = Bar()

    @override
    def next_data(self) -> BarData:
        ts = self._current_ts
        result = BarData(ts.to_pydatetime())
        for symbol in self._subscribed_symbols.symbols():
            data: pd.Series = self._data.loc[(symbol, ts)]  # type: ignore
            if not isinstance(data, pd.Series):
                raise ValueError(f"Non unique ({symbol}, {ts})")
            bar = Bar(
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                volume=data["Volume"],
            )
            result[symbol] = bar
        try:
            self._current_ts = next(self._timestamp_iter)  # type: ignore
        except StopIteration:
            self._is_running = False

        return result

class HistoricalFeed(Feed):
    """
    Replays historical data from a HistoryProvider instance as if it were live data.
    """
    def __init__(
        self,
        provider: HistoryProvider,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbols: Optional[list[str]] = None,
        adjusted: bool = True
    ) -> None:
        """Historical feed initializer

        Parameters
        ----------
        provider
            An instance of a class implementing HistoryProvider (e.g., FileHistoryProvider)
        start, optional
            When to start the replay
        end, optional
            When to end the replay
        symbols, optional
            List of symbols to load
        adjusted, optional
            Whether to use adjusted history (default True)
        """
        self._provider = provider
        self._data = pd.DataFrame()
        self._available_symbols = set()
        self._subscribed_symbols = BarData()
        self._timestamp_iter = None

        self._current_ts = pd.Timestamp(year=1970, month=1, day=1, tz=ZoneInfo("America/New_York"))
        self._is_running = False

        self._start_time = start
        self._end_time = end
        self._symbols = symbols
        self._adjusted = adjusted

    @property
    def df(self) -> pd.DataFrame:
        return self._data

    @override
    def start(self) -> None:
        self._data = self._provider.load(
            start=self._start_time,
            end=self._end_time,
            symbols=self._symbols,
            adjusted=self._adjusted
        )

        self._available_symbols = set(self._data.index.get_level_values(0).unique())
        self._timestamp_iter = iter(self._data.index.get_level_values(1).unique())
        self._is_running = True
        try:
            self._current_ts = next(self._timestamp_iter)
            self._is_running = True
        except StopIteration:
            self._is_running = False

    @override
    def stop(self) -> None:
        self._is_running = False

    @override
    def is_running(self) -> bool:
        return self._is_running

    @override
    def subscribe(self, symbol: str) -> None:
        if symbol not in self._available_symbols:
            raise ValueError(f"Symbol {symbol} not available!")
        self._subscribed_symbols[symbol] = Bar()

    @override
    def next_data(self) -> BarData:
        ts = self._current_ts
        result = BarData(ts.to_pydatetime())
        for symbol in self._subscribed_symbols.symbols():
            try:
                data: pd.Series = self._data.loc[(symbol, ts)]
                if not isinstance(data, pd.Series):
                    raise ValueError(f"Non unique ({symbol}, {ts})")
                bar = Bar(
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    volume=data["Volume"],
                )
                result[symbol] = bar
            except KeyError:
                continue

        try:
            self._current_ts = next(self._timestamp_iter) 
        except StopIteration:
            self._is_running = False

        return result
    
class AlpacaLiveStockFeed(Feed):
    """
    A live stock feed that polls the Alpaca API synchronously (no threads/async).
    Uses the Historical Data Client to get the latest minute bar repeatedly.
    """
    def __init__(self):
        super().__init__()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_API_SECRET")

        if not api_key or not secret_key:
            logger.error("API keys missing! Check environment variables.")
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set in environment variables.")

        self._data_client = ad.StockHistoricalDataClient(api_key, secret_key)
        self._is_running = False
        self._subscribed_symbols = set()
        self._last_timestamp = None
        self._poll_interval = 5
        self._market_tz = ZoneInfo("America/New_York")


    @override
    def start(self) -> None:
        if self._is_running:
            return
        self._is_running = True
        logger.info("Alpaca polling feed started.")

    @override
    def stop(self) -> None:
        self._is_running = False
        logger.info("Alpaca polling feed stopped.")

    @override
    def is_running(self) -> bool:
        return self._is_running

    @override
    def subscribe(self, symbol: str) -> None:
        self._subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to {symbol} for polling live data.")


    @override
    def next_data(self) -> BarData:
        if not self._is_running:
            raise RuntimeError("Feed is not running. Call start() first.")

        if not self._subscribed_symbols:
            logger.warning("next_data called with no subscribed symbols. Waiting...")
            time.sleep(self._poll_interval)

        while True:
            now = datetime.now(self._market_tz)
            start_time = now - timedelta(minutes=5)

            bar_request = ad.StockBarsRequest(
                symbol_or_symbols=list(self._subscribed_symbols),
                timeframe=ad.TimeFrame(amount=1, unit=ad.TimeFrameUnit.Minute),
                feed=ad.DataFeed.IEX,
                start=start_time
            )

            bars = self._data_client.get_stock_bars(bar_request)

            if bars is None or bars.df.empty:
                logger.debug("No data returned from poll, sleeping...")
                time.sleep(self._poll_interval)
                continue

        # Find the latest timestamp each symbol has data for
            symbol_latest = {}
            all_present = True
            for symbol in self._subscribed_symbols:
                try:
                    symbol_latest[symbol] = bars.df.loc[symbol].index.max()
                except KeyError:
                    logger.warning(f"Symbol {symbol} has no data in this poll, skipping.")
                    all_present = False
                    break

            if not all_present:
                time.sleep(self._poll_interval)
                continue

        # Use the most conservative (oldest) latest timestamp across symbols
            latest_timestamp = min(symbol_latest.values())

            if self._last_timestamp is not None and latest_timestamp <= self._last_timestamp:
                logger.debug(f"Waiting for new bar. Last processed: {self._last_timestamp}")
                time.sleep(self._poll_interval)
                continue

        # Build result — normalise timestamp to UTC for consistent lookups
            latest_timestamp_utc = latest_timestamp.tz_convert("UTC")
            result = BarData(latest_timestamp_utc.to_pydatetime())

            lookup_ok = True
            for symbol in self._subscribed_symbols:
                try:
                    bar_series = bars.df.loc[symbol].loc[latest_timestamp]
                    result[symbol] = Bar(
                        open=bar_series['open'],
                        high=bar_series['high'],
                        low=bar_series['low'],
                        close=bar_series['close'],
                        volume=bar_series['volume'],
                    )
                except KeyError:
                    logger.warning(f"Lookup failed for {symbol} at {latest_timestamp}, retrying.")
                    lookup_ok = False
                    break

            if not lookup_ok:
                time.sleep(self._poll_interval)
                continue

            self._last_timestamp = latest_timestamp
            return result

        logger.debug(f"Waiting for new bar. Last processed: {self._last_timestamp}")
        time.sleep(self._poll_interval)
