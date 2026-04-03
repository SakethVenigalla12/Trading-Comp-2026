class Position:
    """Represents some asset amount currently invested"""

    def __init__(self, symbol: str, qty: float) -> None:
        self.symbol = symbol
        self.qty = qty

    def value(self, price: float) -> float:
        """Market value of position given price"""
        return self.qty * price

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Position):
            return False
        return self.symbol == value.symbol and self.qty == value.qty
