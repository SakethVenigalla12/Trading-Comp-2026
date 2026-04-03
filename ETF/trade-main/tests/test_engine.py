from datetime import datetime
from typing import override
from unittest import mock

from systrade.broker import BacktestBroker, Broker
from systrade.data import Bar, BarData, ExecutionReport
from systrade.engine import Engine
from systrade.feed import Feed
from systrade.strategy import Strategy


class FakeFeed(Feed):
    def __init__(self, data: list[BarData]) -> None:
        """Initialize with bar events in ascending order"""
        super().__init__()
        self._data = data.copy()

    @override
    def start(self) -> None:
        # no-op
        pass

    @override
    def stop(self) -> None:
        # no-op
        pass

    @override
    def is_running(self) -> bool:
        # While we still have data left
        return bool(self._data)

    @override
    def subscribe(self, symbol: str) -> None:
        # no-op
        pass

    @override
    def next_data(self) -> BarData:
        """Block until returning the next available data for subscribed
        symbols"""
        return self._data.pop(0)


class FakeStrategy(Strategy):
    """A fake buy and hold strategy for testing"""

    def __init__(self, sym: str, qty: float) -> None:
        super().__init__()
        self.sym = sym
        self.qty = qty
        self.exec_report: ExecutionReport | None = None

    @override
    def on_start(self) -> None:
        self.subscribe(self.sym)

    @override
    def on_data(self, data: BarData) -> None:
        if not self.portfolio.is_invested():
            if data.get(self.sym):
                self.post_market_order(self.sym, self.qty)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        self.exec_report = report


def test_engine_run_setup():
    """Test that engine sets up things correctly before feed starts. This
    example uses mocks"""
    feed: mock.Mock = mock.create_autospec(Feed)
    broker: mock.Mock = mock.create_autospec(Broker)
    strategy: mock.Mock = mock.create_autospec(Strategy)

    # Feed will never actually start
    feed.is_running.side_effect = [False]
    engine = Engine(feed, broker, strategy, cash=1000)
    engine.run()

    feed.start.assert_called()
    strategy.setup_context.assert_called()
    strategy.on_start.assert_called()


def test_engine_round_trip():
    """Test an engine loop using using fakes. This one is a bit more like an
    integration test since it tests everything together."""

    # Initial bar where strategy can make decision whether to submit order
    d1 = BarData(datetime(2025, 1, 1))
    # Second bar when order is filled
    d2 = BarData(datetime(2025, 1, 1))
    sym = "ABC"
    qty = 10
    d1[sym] = Bar(close=123)
    d2[sym] = Bar(close=124)

    feed = FakeFeed([d1, d2])
    broker = BacktestBroker()
    strategy = FakeStrategy(sym, qty)

    engine = Engine(feed, broker, strategy, cash=1000)
    engine.run()

    # High level checks that we made it through both loops
    assert strategy.exec_report is not None
    assert engine.portfolio.is_invested_in(sym)
