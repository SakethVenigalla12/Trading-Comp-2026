from datetime import datetime

from systrade.broker import BacktestBroker
from systrade.data import Bar, BarData, ExecutionReport, Order, OrderType


def _make_market_order(sym: str, qty: float, order_id: str) -> Order:
    """Helper to make market order objects"""
    return Order(
        id=order_id,
        symbol=sym,
        quantity=qty,
        type=OrderType.MARKET,
        submit_time=datetime.now(),
    )


def test_backtest_broker_fills_market_orders_at_next_bar():
    """Ensure market orders get completely filled on next available bar"""
    broker = BacktestBroker()
    sym = "ABC"
    ord1 = _make_market_order(sym, 5, "O1")
    ord2 = _make_market_order(sym, 10, "O2")

    broker.post_order(ord1)
    broker.post_order(ord2)

    data = BarData(as_of=datetime(2025, 1, 1))
    data[sym] = Bar(open=30, close=35)
    broker.on_data(data)
    rep1, rep2 = broker.pop_latest()

    def compare(rep: ExecutionReport, ord: Order):
        assert rep.order == ord
        # We should be getting filled at the open
        assert rep.last_price == data[sym].open
        # We should be getting completely filled on market orders
        assert rep.last_quantity == ord.quantity
        assert rep.cum_quantity == ord.quantity
        assert rep.rem_quantity == 0
        # As of time should match the bar timestamp
        assert rep.fill_timestamp == data.as_of

    # Compare respective fields
    compare(rep1, ord1)
    compare(rep2, ord2)

    # All reports are cleared out
    assert not broker.pop_latest()
