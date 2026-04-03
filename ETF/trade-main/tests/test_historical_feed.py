# tests/test_historical_feed_pytest.py

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from systrade.data import Bar
# Import the new classes: HistoricalFeed and FileHistoryProvider
from systrade.feed import HistoricalFeed
from systrade.history import FileHistoryProvider


# --- Helper functions (from your example) ---

def _extract_raw_bar_data(df: pd.DataFrame, date_obj: datetime, sym: str) -> pd.Series:
    """Extracts raw bar from a DataFrame in Series form (handles datetime objects now)"""
    # Adjust extraction for timezone-aware Date column in our new implementation
    target_date = date_obj.date().isoformat()
    # Need to access index levels to filter correctly on the multi-index DF we use internally
    filtered_df = df.reset_index() 
    return filtered_df[filtered_df["Symbol"].eq(sym) & filtered_df["Date"].dt.date.eq(date_obj.date())].iloc[0]


def _assert_bar_eq_raw(bar1: Bar, raw: pd.Series, tol=1e-6) -> None:
    """Helper function to check that bar data matches series that would have produced it"""
    assert bar1.open == pytest.approx(raw["Open"], abs=tol)
    assert bar1.high == pytest.approx(raw["High"], abs=tol)
    assert bar1.low == pytest.approx(raw["Low"], abs=tol)
    assert bar1.close == pytest.approx(raw["Close"], abs=tol)
    assert bar1.volume == pytest.approx(raw["Volume"], abs=tol)

# --- Tests for HistoricalFeed using FileHistoryProvider as concrete dependency ---

def test_is_running_historical_feed():
    """Test is_running indicators for HistoricalFeed"""
    data_path = Path(__file__).parent / "bars.csv"
    provider = FileHistoryProvider(data_path)
    feed = HistoricalFeed(provider)
    
    assert not feed.is_running()
    feed.start()
    assert feed.is_running()
    feed.stop()
    assert not feed.is_running()


def test_subscribe_if_no_symbol_throws_hf():
    """Test subscription will raise exception if no data"""
    data_path = Path(__file__).parent / "bars.csv"
    provider = FileHistoryProvider(data_path)
    feed = HistoricalFeed(provider)
    feed.start()

    with pytest.raises(ValueError):
        # MSFT is not in bars.csv
        feed.subscribe("MSFT")


def test_next_data_empty_sub_hf():
    """Test next_data returns nothing if not subscribed"""
    data_path = Path(__file__).parent / "bars.csv"
    provider = FileHistoryProvider(data_path)
    feed = HistoricalFeed(provider)
    feed.start()

    # next_data returns a BarData object, which should be empty if nothing is subscribed
    assert not feed.next_data().symbols()


def test_next_data_after_sub_hf():
    """Test next_data returns data if subscribed"""
    data_path = Path(__file__).parent / "bars.csv"
    provider = FileHistoryProvider(data_path)
    feed = HistoricalFeed(provider)
    feed.start()
    feed.subscribe("NVDA")

    # next_data returns a BarData object which should now have symbols
    assert "NVDA" in feed.next_data().symbols()


def test_start_with_date_filter_hf():
    """Test date filters work when passed to the HistoricalFeed constructor."""
    data_path = Path(__file__).parent / "bars.csv"
    
    start_date = datetime(2005, 2, 4, tzinfo=ZoneInfo("America/New_York"))
    end_date = datetime(2005, 2, 8, tzinfo=ZoneInfo("America/New_York"))

    # Pass start/end dates to the HistoricalFeed, which passes them to the Provider.load()
    provider = FileHistoryProvider(data_path)
    feed = HistoricalFeed(provider, start=start_date, end=end_date)
    
    feed.start()
    feed.subscribe("NVDA")

    assert feed.is_running()
    d1 = feed.next_data()
    d2 = feed.next_data()
    d3 = feed.next_data()
    
    # After the 3rd data point, the iterator runs out
    assert not feed.is_running()
    
    assert d1.as_of.date() == date.fromisoformat("2005-02-04")
    assert d2.as_of.date() == date.fromisoformat("2005-02-07")
    assert d3.as_of.date() == date.fromisoformat("2005-02-08")


def test_next_data_populates_bars_on_subscription_hf():
    """Test single subscription data is populated appropriately in bar data."""
    source_path = Path(__file__).parent / "bars2.csv"
    sym = "GS"
    df_raw = pd.read_csv(source_path) # Raw DF for comparison
    
    provider = FileHistoryProvider(source_path)
    feed = HistoricalFeed(provider)
    feed.start()
    feed.subscribe(sym)
    
    # Get the internal multi-index dataframe the feed is using after start()
    feed_df = feed.df 

    assert feed.is_running()
    d1 = feed.next_data()
    d2 = feed.next_data()

    assert sym in d1.symbols()
    assert sym in d2.symbols()

    bar1 = d1[sym]
    bar2 = d2[sym]

    # Use helper to compare the produced Bar object against the raw CSV data
    _assert_bar_eq_raw(bar1, _extract_raw_bar_data(df_raw, d1.as_of, sym))
    _assert_bar_eq_raw(bar2, _extract_raw_bar_data(df_raw, d2.as_of, sym))