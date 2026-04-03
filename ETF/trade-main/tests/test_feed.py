from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from systrade.data import Bar
from systrade.feed import FileFeed


def _extract_raw_bar_data(df: pd.DataFrame, date: datetime, sym: str) -> pd.Series:
    """Extracts raw bar from a DataFrame in Series form"""
    return df[df["Symbol"].eq(sym) & df["Date"].eq(date.date().isoformat())].iloc[0]


def _assert_bar_eq_raw(bar1: Bar, raw: pd.Series, tol=1e-6) -> None:
    """Helper function to check that bar data matches series that would have
    produced it"""
    assert bar1.open == pytest.approx(raw["Open"], abs=tol)
    assert bar1.high == pytest.approx(raw["High"], abs=tol)
    assert bar1.low == pytest.approx(raw["Low"], abs=tol)
    assert bar1.close == pytest.approx(raw["Close"], abs=tol)
    assert bar1.volume == pytest.approx(raw["Volume"], abs=tol)


def test_is_running():
    """Test is_running indicators"""
    feed = FileFeed(Path(__file__).parent / "bars.csv")
    assert not feed.is_running()
    feed.start()
    assert feed.is_running()
    feed.stop()
    assert not feed.is_running()


def test_subscribe_if_no_symbol_throws():
    """Test subscription will raise exception if no data"""
    feed = FileFeed(Path(__file__).parent / "bars.csv")
    feed.start()

    with pytest.raises(ValueError):
        feed.subscribe("MSFT")


def test_next_data_empty_sub():
    """Test next_data returns nothing if not subscribed"""
    feed = FileFeed(Path(__file__).parent / "bars.csv")
    feed.start()

    assert not feed.next_data()


def test_next_data_after_sub():
    """Test next_data returns nothing if not subscribed"""
    feed = FileFeed(Path(__file__).parent / "bars.csv")
    feed.start()
    feed.subscribe("NVDA")

    assert feed.next_data()


def test_start_with_date_filter():
    """Test date filters work"""
    feed = FileFeed(
        Path(__file__).parent / "bars.csv", start="2005-02-04", end="2005-02-08"
    )
    feed.start()
    feed.subscribe("NVDA")

    assert feed.is_running()
    d1 = feed.next_data()
    d2 = feed.next_data()
    d3 = feed.next_data()
    assert not feed.is_running()
    assert d1.as_of.date() == date.fromisoformat("2005-02-04")
    assert d2.as_of.date() == date.fromisoformat("2005-02-07")
    assert d3.as_of.date() == date.fromisoformat("2005-02-08")


def test_next_data_populates_bars_on_subscription():
    """Test single subscription data is populated appropriately in bar data
    (added in solution)"""
    source = Path(__file__).parent / "bars2.csv"
    sym = "GS"
    df = pd.read_csv(source)
    feed = FileFeed(source)
    feed.start()
    feed.subscribe(sym)

    assert feed.is_running()
    d1 = feed.next_data()
    d2 = feed.next_data()

    assert sym in d1.symbols()
    assert sym in d2.symbols()

    bar1 = d1[sym]
    bar2 = d2[sym]

    _assert_bar_eq_raw(bar1, _extract_raw_bar_data(df, d1.as_of, sym))
    _assert_bar_eq_raw(bar2, _extract_raw_bar_data(df, d2.as_of, sym))


def test_next_data_populates_bars_on_subscription_dual_sub():
    """Test multiple subscription data is populated appropriately in bar data
    (added in solution)"""
    source = Path(__file__).parent / "bars2.csv"
    sym1 = "GS"
    sym2 = "NVDA"
    df = pd.read_csv(source)
    feed = FileFeed(source)
    feed.start()
    feed.subscribe(sym1)
    feed.subscribe(sym2)

    assert feed.is_running()
    d1 = feed.next_data()

    assert sym1 in d1.symbols()
    assert sym2 in d1.symbols()

    bar1_1 = d1[sym1]
    bar1_2 = d1[sym2]

    _assert_bar_eq_raw(bar1_1, _extract_raw_bar_data(df, d1.as_of, sym1))
    _assert_bar_eq_raw(bar1_2, _extract_raw_bar_data(df, d1.as_of, sym2))
