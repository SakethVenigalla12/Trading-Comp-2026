from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, override
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import psycopg 
from psycopg.rows import dict_row

class HistoryProvider(ABC):
    """Historical data loader abstract base class."""

    @abstractmethod
    def load(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbols: Optional[list[str]] = None,
        adjusted: bool = True 
    ) -> pd.DataFrame:
        """
        Load historical data from storage, returning a DataFrame with schema
        corresponding to systrade.data.Bar.

        Args:
            start (datetime, optional): Start date for data.
            end (datetime, optional): End date for data.
            symbols (list[str], optional): List of symbols to retrieve.
            adjusted (bool): Whether to return data adjusted for corporate actions.
        """
        pass

class FileHistoryProvider(HistoryProvider):
    """Loads historical data from a local file (e.g., CSV)."""

    def __init__(self, path: str | Path, timezone_str: str = "America/New_York"):
        """
        Initializes the provider with the path to the data file and a timezone string.
        """
        self._path = path
        try:
            self._tz = ZoneInfo(timezone_str)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone: {timezone_str}")

    @override
    def load(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbols: Optional[list[str]] = None,
        adjusted: bool = True
    ) -> pd.DataFrame:
        
        df = pd.read_csv(self._path)
        
        df["Date"] = pd.to_datetime(df["Date"])
        if df["Date"].dt.tz is None:
            df["Date"] = df["Date"].dt.tz_localize(self._tz, ambiguous='NaT', nonexistent='NaT')
        else:
            df["Date"] = df["Date"].dt.tz_convert(self._tz)
        
 
        df["Symbol"] = df["Symbol"].astype(pd.StringDtype(storage="python"))
        df = df.set_index(["Symbol", "Date"]).sort_index()

        if start is not None:
            start_localized = start.astimezone(self._tz) if start.tzinfo else start.replace(tzinfo=self._tz)
            df = df.loc[df.index.get_level_values('Date') >= start_localized]
            
        if end is not None:
            end_localized = end.astimezone(self._tz) if end.tzinfo else end.replace(tzinfo=self._tz)
            df = df.loc[df.index.get_level_values('Date') <= end_localized]

        if symbols is not None:
            df = df.loc[pd.IndexSlice[symbols, :], :]
            
        if adjusted:
            print("Note: FileHistoryProvider assumes input file is already adjusted when adjusted=True is requested.")
        
        return df

class QuestDBHistoryProvider(HistoryProvider):
    """Loads historical data from a QuestDB instance using psycopg."""

    def __init__(self, connection_url: str, timezone_str: str = "America/New_York"):
        """
        Initializes the provider with a QuestDB connection URL 
        (e.g., 'postgresql://user:pass@host:port/db').
        """
        self._connection_url = connection_url
        try:
            self._tz = ZoneInfo(timezone_str)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone: {timezone_str}")

    @override
    def load(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbols: Optional[list[str]] = None,
        adjusted: bool = True
    ) -> pd.DataFrame:
        
        query_sql, query_params = self._build_query(start, end, symbols, adjusted)
        
        with psycopg.connect(self._connection_url) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query_sql, query_params)
                records = cur.fetchall()

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.rename(columns={'date': 'Date', 'open': 'Open', 'high': 'High', 
                                'low': 'Low', 'close': 'Close', 'volume': 'Volume',
                                'symbol': 'Symbol'})
        
        df["Symbol"] = df["Symbol"].astype(pd.StringDtype(storage="python"))
        df = df.set_index(["Symbol", "Date"]).sort_index()
        
        return df

    def _build_query(self, start, end, symbols, adjusted: bool) -> tuple[str, dict]:
        """Constructs the SQL query using psycopg parameter binding."""
        
        params = {}
        where_clause_parts = []

        if start:
            where_clause_parts.append("date >= %(start_date)s")
            params['start_date'] = start
        if end:
            where_clause_parts.append("date <= %(end_date)s")
            params['end_date'] = end

        if symbols:
            where_clause_parts.append("symbol IN %(symbols)s")
            params['symbols'] = tuple(symbols)

        where_sql = "WHERE " + " AND ".join(where_clause_parts) if where_clause_parts else ""
        
        if not adjusted:
            sql_query = f"SELECT symbol, date, open, high, low, close, volume FROM equity_day_bars {where_sql} ORDER BY date ASC"
        else:
            sql_query = f"SELECT symbol, date, open, high, low, close, volume FROM equity_day_bars {where_sql} ORDER BY date ASC"
            print("Warning: 'adjusted=True' requested, but complex adjustment SQL not fully implemented yet.")

        return sql_query, params
