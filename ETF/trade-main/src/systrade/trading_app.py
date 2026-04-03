"""Systematic Trading Application"""

import os
import json
import logging
from datetime import datetime
from typing import override
from zoneinfo import ZoneInfo

from systrade.feed import AlpacaLiveStockFeed
from systrade.broker import AlpacaBroker
from systrade.strategy import Strategy
from systrade.data import BarData, ExecutionReport
from systrade.engine import Engine

import math
# ---------------------------
# --------- LOGGING ---------
# ----- logging imports -----
import logging.config
import logging.handlers
import json
import pathlib
# instantiate logger
logger = logging.getLogger(__name__)

# --- LOGGER CONFIG ---
# Verbose dictionary-type config
#+for custom logger.
# Config file found in:
# /config/logger/config.json
# (source: youtube.com/mCoding)
def setup_logging():
    config_file = pathlib.Path("config/logger/config.json")
    with open(config_file) as f_in:
        config = json.load(f_in)
    logging.config.dictConfig(config)
# initialize escape codes for
#+color-coding logs
red = "\033[31m"
green = "\033[32m"
yellow = "\033[33m"
blue = "\033[34m"
hl_red = "\033[41m"
hl_green = "\033[42m"
hl_yellow = "\033[43m"
hl_blue = "\033[44m"
reset = "\033[0m"
# ---------------------

# ===============================================
#        ---- Buy and Hold Strategy ----
# ===============================================
# It's in the name. _It will not sell_.
# Could also be called the diamond hands strategy
class LongStrategy(Strategy):
    """
    Buy and hold. "Go long" strategy
    """
    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.symbol = symbol
        self.history: list[float] = []
        self.trading_records: list[dict] = []
        logger.info(f"Long Strategy initialized for {self.symbol}")

    @override
    def on_start(self) -> None:
        """Subscribe to the symbol on strategy start"""
        self.subscribe(self.symbol)

    # this will just buy when it gets its first price
    @override
    def on_data(self, data: BarData) -> None:
        """Processes incoming 1-minute bars live."""
        self.current_time = data.as_of

        if self.symbol in data.symbols():
            bar = data[self.symbol]
            price = bar.close

            logger.info(f"Processing bar for {self.symbol} at {data.as_of}: Close={price}")

            # 30% buffer for daytrading
            #-------------------------
            # If you are marked by alpaca as a pattern daytrader,
            #+they will nerf your buying power so this is added
            #+to skirt that.
            qty = math.floor((self.portfolio.buying_power() * 0.70) / price)
            if qty > 0:
                self.post_market_order(self.symbol, quantity=qty)
                logger.info(f"{hl_green}Buy signal! Posting market order for {qty} shares of {self.symbol}{reset}")
                self.order_pending = True
                self._record_trade("BUY", qty, price)
            else:
                logger.warning(f"{yellow}Quantity calculated as 0. Buying Power: {self.portfolio.buying_power()}{reset}")

            
            # add price to tracking log
            self.history.append(price)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        """Called on an order update"""
        log_report = report.__dict__.copy()
        log_report['fill_timestamp_iso'] = report.fill_timestamp.isoformat()
        logger.info(f"Notified of execution: {log_report}")
        self.trading_records.append(log_report)

    # this function records trades into a json file
    # the trades can be extracted out of a dockerized
    #+instance of this trading app with the following
    #+command: $ docker cp <container_name_or_id>:trading_results.json trading_results.json
    def _record_trade(self, side, qty, price):
        """Helper to save a simple record locally."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': side,
            'quantity': qty,
            'price': price
        }
        with open("trading_results.json", "a") as f:
            f.write(json.dumps(record) + "\n")

# =============================================
#    ------- Stat Arb Strategy -----------
# =============================================
# Pairs trading strategy between two correlated ETFs.
# Uses a rolling OLS hedge ratio and z-score to detect
# when the spread between the two assets has deviated
# from its historical mean, then trades the reversion.
#
# Long spread  (z too low):  buy symbol_a, short symbol_b
# Short spread (z too high): short symbol_a, buy symbol_b

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

class StatArbStrategy(Strategy):
    """
    Statistical arbitrage (pairs trading) strategy.
    Trades the spread between two correlated symbols.
    """
    def __init__(
        self,
        symbol_a: str,          # e.g. "SPY"
        symbol_b: str,          # e.g. "QQQ"
        window: int = 10,       # rolling lookback window (bars)
        entry_z: float = 2.0,   # z-score to enter a trade
        exit_z: float = 0.5,    # z-score to exit a trade
    ) -> None:
        super().__init__()
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

        self.prices_a: list[float] = []
        self.prices_b: list[float] = []
        self.trading_records: list[dict] = []

        # "long" = long A short B | "short" = short A long B | None = flat
        self.spread_position = None

        logger.info(
            f"StatArb Strategy initialized: {symbol_a}/{symbol_b} "
            f"window={window}, entry_z={entry_z}, exit_z={exit_z}"
        )

    @override
    def on_start(self) -> None:
        """Subscribe to both symbols on strategy start"""
        self.subscribe(self.symbol_a)
        self.subscribe(self.symbol_b)

    @override
    def on_data(self, data: BarData) -> None:
        """Processes incoming bars live."""
        self.current_time = data.as_of

        # Wait until we have data for both symbols
        if self.symbol_a not in data.symbols() or self.symbol_b not in data.symbols():
            return

        price_a = data[self.symbol_a].close
        price_b = data[self.symbol_b].close

        logger.info(
            f"Processing bar at {data.as_of}: "
            f"{self.symbol_a}={price_a:.2f}, {self.symbol_b}={price_b:.2f}"
        )

        self.prices_a.append(price_a)
        self.prices_b.append(price_b)

        # Need enough history before trading
        if len(self.prices_a) < self.window:
            logger.debug(f"Building history: {len(self.prices_a)}/{self.window} bars")
            return

        # Rolling window arrays
        pa = np.array(self.prices_a[-self.window:])
        pb = np.array(self.prices_b[-self.window:])

        # Estimate hedge ratio via OLS
        beta = OLS(pa, add_constant(pb)).fit().params[1]

        # Define our objective functions: spread and z-score
        # We assume the two are correlated and can be expressed as a "LC" of eachother 
        # Beta here beta represents a sort of "hedge ratio" (how much pa tends to move per unit of pb)
        spread = pa - beta * pb
        spread_mean = spread.mean()
        spread_std = spread.std()

       

        if spread_std == 0:
            return

        current_spread = price_a - beta * price_b
        z_score = (current_spread - spread_mean) / spread_std

        logger.debug(
            f"Spread z-score={z_score:.2f}, beta={beta:.4f}, "
            f"position={self.spread_position}"
        )

        # --- Exit logic ---
        if self.spread_position == "long" and abs(z_score) < self.exit_z:
            pos_a = self.portfolio.position(self.symbol_a)
            pos_b = self.portfolio.position(self.symbol_b)
            if pos_a:
                self.post_market_order(self.symbol_a, quantity=-pos_a.qty)
                self._record_trade("SELL", pos_a.qty, price_a, self.symbol_a)
            if pos_b:
                self.post_market_order(self.symbol_b, quantity=-pos_b.qty)
                self._record_trade("BUY", abs(pos_b.qty), price_b, self.symbol_b)
            self.spread_position = None
            logger.info(f"{hl_green}EXIT long spread | z={z_score:.2f}{reset}")

        elif self.spread_position == "short" and abs(z_score) < self.exit_z:
            pos_a = self.portfolio.position(self.symbol_a)
            pos_b = self.portfolio.position(self.symbol_b)
            if pos_a:
                self.post_market_order(self.symbol_a, quantity=-pos_a.qty)
                self._record_trade("BUY", abs(pos_a.qty), price_a, self.symbol_a)
            if pos_b:
                self.post_market_order(self.symbol_b, quantity=-pos_b.qty)
                self._record_trade("SELL", pos_b.qty, price_b, self.symbol_b)
            self.spread_position = None
            logger.info(f"{hl_green}EXIT short spread | z={z_score:.2f}{reset}")

        # --- Entry logic ---
        elif self.spread_position is None:
            # Use 70% of buying power split across two legs (matching your day trading buffer)
            half_bp = self.portfolio.buying_power() * 0.70 / 2

            if z_score < -self.entry_z:
                # Spread too low: buy A, short B
                qty_a = math.floor(half_bp / price_a)
                qty_b = math.floor(half_bp / price_b)
                if qty_a > 0 and qty_b > 0:
                    self.post_market_order(self.symbol_a, quantity=qty_a)
                    self.post_market_order(self.symbol_b, quantity=-qty_b)
                    self.spread_position = "long"
                    self._record_trade("BUY", qty_a, price_a, self.symbol_a)
                    self._record_trade("SELL", qty_b, price_b, self.symbol_b)
                    logger.info(f"{hl_green}ENTER long spread | z={z_score:.2f} | "
                                f"Buy {qty_a} {self.symbol_a} @ {price_a:.2f}, "
                                f"Short {qty_b} {self.symbol_b} @ {price_b:.2f}{reset}")
                else:
                    logger.warning(f"{yellow}Insufficient buying power to enter spread{reset}")

            elif z_score > self.entry_z:
                # Spread too high: short A, buy B
                qty_a = math.floor(half_bp / price_a)
                qty_b = math.floor(half_bp / price_b)
                if qty_a > 0 and qty_b > 0:
                    self.post_market_order(self.symbol_a, quantity=-qty_a)
                    self.post_market_order(self.symbol_b, quantity=qty_b)
                    self.spread_position = "short"
                    self._record_trade("SELL", qty_a, price_a, self.symbol_a)
                    self._record_trade("BUY", qty_b, price_b, self.symbol_b)
                    logger.info(f"{hl_red}ENTER short spread | z={z_score:.2f} | "
                                f"Short {qty_a} {self.symbol_a} @ {price_a:.2f}, "
                                f"Buy {qty_b} {self.symbol_b} @ {price_b:.2f}{reset}")
                else:
                    logger.warning(f"{yellow}Insufficient buying power to enter spread{reset}")

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        """Called on an order update"""
        log_report = report.__dict__.copy()
        log_report['fill_timestamp_iso'] = report.fill_timestamp.isoformat()
        logger.info(f"Notified of execution: {log_report}")
        self.trading_records.append(log_report)

    def _record_trade(self, side, qty, price, symbol):
        """Helper to save a simple record locally."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': qty,
            'price': price
        }
        with open("trading_results.json", "a") as f:
            f.write(json.dumps(record) + "\n")
    
# =============================================
#    -------  Momentum strategy -----------
# =============================================
# This is the strategy that's most developed in 
#+the repo. You can edit this one for ease, or 
#+anything else to your liking. Just make sure
#+it runs.


class MomentumStrategy(Strategy):
    """
    Momentum strategy with long/short support.
    """
    def __init__(self, symbol: str) -> None:
        super().__init__()
        self.symbol = symbol
        self.history: list[float] = []
        self.trading_records: list[dict] = []
        logger.info(f"Momentum Strategy initialized for {self.symbol}")

    @override
    def on_start(self) -> None:
        """Subscribe to the symbol on strategy start"""
        self.subscribe(self.symbol)

    # main logic that handles buying and selling for this
    #+moment strategy. there's almost surely a better way to
    #+both code and format this. LOLL
    # i am partial to thinking the bad logic offers nice
    #+logging though....
    @override
    def on_data(self, data: BarData) -> None:
        """Processes incoming 1-minute bars live."""
        self.current_time = data.as_of

        if self.symbol in data.symbols():
            bar = data[self.symbol]
            price = bar.close

            logger.info(f"Processing bar for {self.symbol} at {data.as_of}: Close={price}")

            if len(self.history) >= 2:
                buy_signal = price > self.history[-1] > self.history[-2]
                sell_signal = price < self.history[-1] < self.history[-2]

                holding = self.portfolio.is_invested_in(self.symbol)

                # this block will open a long position
                if buy_signal and not holding:
                    logger.debug(f"{blue}Buying Power={self.portfolio.buying_power()}, Invested={self.portfolio.is_invested_in(self.symbol)}{reset}")
                    # add 5% buying power buffer
                    #add temp 30% buffer for daytrading or something
                    qty = math.floor((self.portfolio.buying_power() * 0.70) / price)
                    if qty > 0:
                        self.post_market_order(self.symbol, quantity=qty)
                        logger.info(f"{hl_green}Buy signal! Posting market order for {qty} shares of {self.symbol}{reset}")
                        self.order_pending = True
                        self._record_trade("BUY", qty, price)
                    else:
                        logger.warning(f"{yellow}Quantity calculated as 0. Buying Power: {self.portfolio.buying_power()}{reset}")

                # this block will open a short position
                elif sell_signal and not holding:
                    logger.debug(f"{blue}Buying Power={self.portfolio.buying_power()}, Invested={self.portfolio.is_invested_in(self.symbol)}{reset}")
                    # add 5% buying power buffer
                    #add temp 30% buffer for daytrading or something
                    qty = math.floor((self.portfolio.buying_power() * 0.70) / price)
                    if qty > 0:
                        self.post_market_order(self.symbol, quantity=-qty)
                        logger.info(f"{hl_red}Sell signal! Posting market order for {qty} shares of {self.symbol}{reset}")
                        self.order_pending = True
                        self._record_trade("SELL", qty, price)
                    else:
                        logger.warning(f"{yellow}Quantity calculated as 0. Buying Power: {self.portfolio.buying_power()}{reset}")

                # this block will close a short position
                elif buy_signal and holding:
                    logger.debug(f"{blue}Buying Power={self.portfolio.buying_power()}, Invested={self.portfolio.is_invested_in(self.symbol)}{reset}")
                    pos = self.portfolio.position(self.symbol)
                    logger.info(f"{hl_yellow}Buy signal! Closing short position of {pos.qty} shares of {self.symbol}{reset}")
                    self.post_market_order(self.symbol, quantity=pos.qty)
                    self.order_pending = True
                    self._record_trade("BUY", pos.qty, price)

                # this block will close a long position
                elif sell_signal and holding:
                    logger.debug(f"{blue}Buying Power={self.portfolio.buying_power()}, Invested={self.portfolio.is_invested_in(self.symbol)}{reset}")
                    pos = self.portfolio.position(self.symbol)
                    logger.info(f"{hl_blue}Sell signal! Closing long position of {pos.qty} shares of {self.symbol}{reset}")
                    self.post_market_order(self.symbol, quantity=-pos.qty)
                    self.order_pending = True
                    self._record_trade("SELL", pos.qty, price)
            
            # add price to tracking log
            self.history.append(price)

    @override
    def on_execution(self, report: ExecutionReport) -> None:
        """Called on an order update"""
        log_report = report.__dict__.copy()
        log_report['fill_timestamp_iso'] = report.fill_timestamp.isoformat()
        logger.info(f"Notified of execution: {log_report}")
        self.trading_records.append(log_report)

    # this function records trades into a json file
    # the trades can be extracted out of a dockerized
    #+instance of this trading app with the following
    #+command: $ docker cp <container_name_or_id>:trading_results.json trading_results.json
    def _record_trade(self, side, qty, price):
        """Helper to save a simple record locally."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': side,
            'quantity': qty,
            'price': price
        }
        with open("trading_results.json", "a") as f:
            f.write(json.dumps(record) + "\n")

def main():
    setup_logging()
    logger.info("Starting Systrade Live Trading Application...")
    if not os.getenv("ALPACA_API_KEY"):
        logger.error("API keys not set. Exiting.")
        return

    feed = AlpacaLiveStockFeed()
    broker = AlpacaBroker()
    # i'd recommend choosing which strategy to run here
    strategy = StatArbStrategy(symbol_a="XLK", symbol_b="SPY")

    # NOTE: theres almost surely a better way to store this cash info
    #+we could probably just get it from the API... I'm fairly certain
    #+starting_cash is kind of irrelevant since the strategy already 
    #+just gets the buying power from the live account.
    #... but alas...
    starting_cash = 1000000
    engine = Engine(feed=feed, broker=broker, strategy=strategy, cash=starting_cash)

    logger.info("Engine initialized. Starting run...")

    try:
        engine.run()
        logger.info("Engine run completed successfully.")

    except KeyboardInterrupt:
        logger.info("Trading interrupted by user. Stopping engine.")
    except Exception as e:
        logger.error(f"{hl_red}An unexpected error occurred: {e}{reset}")

    logger.info("Application stopped.")


if __name__ == "__main__":
    main()

