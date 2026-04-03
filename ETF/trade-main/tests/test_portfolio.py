from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from systrade.data import Bar, BarData
from systrade.portfolio import Portfolio
from systrade.position import Position


def test_position_with_no_position_raises_error():
    pf = Portfolio(1000)
    with pytest.raises(ValueError):
        pf.position("ABC")


def test_position_that_exists():
    pos = Position("ABC", 123)
    pf = Portfolio(1000, current_positions={pos.symbol: pos})
    assert pf.position("ABC") == pos


def test_value_with_no_positions():
    """Test that values are properly computed when initializing with just
    cash"""
    cash = 1000
    pf = Portfolio(cash)
    assert pf.cash() == cash
    assert pf.value() == cash
    assert pf.asset_value() == 0


def test_value_with_positions():
    """Test that values computations of specific positions are correct along
    with total values"""
    cash = 1000
    pos1 = Position("ABC", 5)
    pos2 = Position("DEF", 10)
    positions = {pos1.symbol: pos1, pos2.symbol: pos2}

    data = BarData()
    data[pos1.symbol] = Bar(close=200.5)
    data[pos2.symbol] = Bar(close=300.75)
    pf = Portfolio(cash, current_positions=positions, current_prices=data)

    assert pf.cash() == cash
    assert pf.asset_value_of(pos1.symbol) == pos1.qty * data[pos1.symbol].close
    assert pf.asset_value_of(pos2.symbol) == pos2.qty * data[pos2.symbol].close
    assert pf.asset_value() == pf.asset_value_of(pos1.symbol) + pf.asset_value_of(
        pos2.symbol
    )
    assert (
        pf.value()
        == pf.asset_value_of(pos1.symbol) + pf.asset_value_of(pos2.symbol) + pf.cash()
    )


def test_value_of_missing_position_throws():
    """Test that an exception is raised if trying to compute value of
    non-invested symbol"""
    pf = Portfolio(1000)
    with pytest.raises(ValueError):
        pf.asset_value_of("ABC")


def test_value_with_no_data_throws():
    """Test that error is raised if we don't have latest prices to compute a
    value"""
    pos = Position("ABC", 5)
    pf = Portfolio(1000, current_positions={pos.symbol: pos})
    with pytest.raises(RuntimeError):
        _ = pf.value()
    with pytest.raises(RuntimeError):
        _ = pf.asset_value()
    with pytest.raises(RuntimeError):
        _ = pf.asset_value_of("ABC")


def test_as_of_uses_bar_data():
    """Test that the portfolio as of date reflects the time of the bar data"""
    as_of = datetime(2025, 1, 2, 1, 1, 1)
    data = BarData(as_of=as_of)
    pf = Portfolio(1000, current_prices=data)
    assert pf.as_of() == as_of


def test_invested_indicators():
    """Test that indicators for whether invested in a position or at all work"""
    pos1 = Position("ABC", 5)
    pos2 = Position("DEF", 10)
    positions = {pos1.symbol: pos1, pos2.symbol: pos2}
    pf1 = Portfolio(1000)
    pf2 = Portfolio(1000, current_positions=positions)

    assert not pf1.is_invested()
    assert pf2.is_invested()
    assert pf2.is_invested_in("ABC")
    assert pf2.is_invested_in("DEF")
    assert not pf2.is_invested_in("IJK")


def test_on_data_updates_portfolio():
    """Test that portfolio value functions reflect latest data"""
    pos = Position("ABC", 5)
    pf = Portfolio(1000, current_positions={pos.symbol: pos})
    data = BarData()

    # Initial update
    data[pos.symbol] = Bar(close=100)
    pf.on_data(data)
    assert pf.asset_value_of(pos.symbol) == pos.qty * data[pos.symbol].close

    # One extra update
    data[pos.symbol] = Bar(close=200.5)
    pf.on_data(data)
    assert pf.asset_value_of(pos.symbol) == pos.qty * data[pos.symbol].close


def test_on_data_updates_activity():
    """Test that portfolio activity is generated with each on_data event"""
    pos = Position("ABC", 5)
    pf = Portfolio(1000, current_positions={pos.symbol: pos})
    data = BarData()

    # Initial update
    data[pos.symbol] = Bar(close=100)
    pf.on_data(data)
    assert pf.asset_value_of(pos.symbol) == pos.qty * data[pos.symbol].close

    # One extra update
    data[pos.symbol] = Bar(close=200.5)
    pf.on_data(data)
    assert pf.asset_value_of(pos.symbol) == pos.qty * data[pos.symbol].close


def test_on_fill_opening_creates_position():
    """Test that a position is generated on fills when one didn't exist
    before"""
    cash = 1000
    sym = "ABC"
    pf = Portfolio(cash)

    # Buy
    fill_price = 100
    fill_qty = 3
    prev_cash = pf.cash()
    pf.on_fill(sym, fill_price, fill_qty)
    assert pf.cash() == prev_cash - fill_price * fill_qty
    assert pf.position(sym) == Position(sym, fill_qty)


def test_on_fill_updating_position():
    """Test that a fill updates an existing position rather than creating a new
    one"""
    cash = 1000
    sym = "ABC"
    pos = Position(sym, 3)
    pf = Portfolio(cash, current_positions={pos.symbol: pos})

    # Buy
    fill_price = 100
    fill_qty = 4
    prev_cash = pf.cash()
    prev_qty = pf.position(sym).qty
    pf.on_fill(sym, fill_price, fill_qty)
    assert pf.cash() == prev_cash - fill_price * fill_qty
    assert pf.position(sym) == Position(sym, prev_qty + fill_qty)

    # Sell
    fill_price = 100
    fill_qty = -2
    prev_cash = pf.cash()
    prev_qty = pf.position(sym).qty
    pf.on_fill(sym, fill_price, fill_qty)
    assert pf.cash() == prev_cash - fill_price * fill_qty
    assert pf.position(sym) == Position(sym, prev_qty + fill_qty)


def test_on_fill_closing_removes_position():
    """Test that a fill updates an existing position rather than creating a new
    one"""
    pos = Position("ABC", 3)
    pf = Portfolio(1000, current_positions={pos.symbol: pos})

    pf.on_fill(pos.symbol, 100, -pos.qty)
    assert not pf.is_invested_in(pos.symbol)


def test_activity_returns_history():
    pf = Portfolio(1000)
    data = BarData(datetime(2025, 1, 1))
    data["ABC"] = Bar(close=50)

    pf.on_data(data)
    pf.on_fill("ABC", 49, 5)

    data = BarData(datetime(2025, 1, 2))
    data["ABC"] = Bar(close=51)
    pf.on_data(data)

    data = BarData(datetime(2025, 1, 3))
    data["ABC"] = Bar(close=48)
    pf.on_data(data)

    pf.on_fill("DEF", 101, 10)
    data = BarData(datetime(2025, 1, 4))
    data["ABC"] = Bar(close=55)
    data["DEF"] = Bar(close=102)
    pf.on_data(data)

    # Would be the result of df.to_records(index=False)
    records = np.rec.array(
        [
            (
                "2025-01-01T00:00:00.000000000",
                1000,
                list([]),
                list([]),
                list([]),
                list([]),
                0,
                1000,
            ),
            (
                "2025-01-02T00:00:00.000000000",
                755,
                list(["ABC"]),
                list([5]),
                list([51]),
                list([255]),
                255,
                1010,
            ),
            (
                "2025-01-03T00:00:00.000000000",
                755,
                list(["ABC"]),
                list([5]),
                list([48]),
                list([240]),
                240,
                995,
            ),
            (
                "2025-01-04T00:00:00.000000000",
                -255,
                list(["ABC", "DEF"]),
                list([5, 10]),
                list([55, 102]),
                list([275, 1020]),
                1295,
                1040,
            ),
        ],
        dtype=[
            ("timestamp", "<M8[ns]"),
            ("cash", "<i8"),
            ("symbols", "O"),
            ("quantities", "O"),
            ("prices", "O"),
            ("asset_values", "O"),
            ("asset_value", "<i8"),
            ("value", "<i8"),
        ],
    )

    pd.testing.assert_frame_equal(
        pf.activity().df(), pd.DataFrame.from_records(records)
    )
