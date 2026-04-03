from systrade.position import Position


def test_value():
    """Test value calculation when provided current price"""
    sym = "ABC"
    price = 123.5
    qty = 10
    pos = Position(sym, qty)

    assert pos.qty == qty
    assert pos.symbol == sym
    assert pos.value(price) == price * qty


def test_eq():
    """Test equal overloaded operator"""
    pos = Position("ABC", 123)
    assert pos == Position("ABC", 123)
    assert pos != Position("DEF", 123)
    assert pos != Position("ABC", 123.1)
    assert pos != ("ABC", 123)
