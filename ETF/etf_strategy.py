class StatArbStrategy(Strategy):
    """
    Statistical arbitrage (pairs trading) strategy.
    Trades the spread between two correlated symbols.
    """
    def __init__(
        self,
        symbol_a: str,          # e.g. "VTI"
        symbol_b: str,          # e.g. "XLK"
        window: int = 390,       # rolling lookback window (minute bars)
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
        if self.spread_position is not None and abs(z_score) > 3.5:
            pos_a = self.portfolio.position(self.symbol_a)
            pos_b = self.portfolio.position(self.symbol_b)
            if pos_a:
                self.post_market_order(self.symbol_a,quantity=-pos_a.qty)
            if pos_b:
                self.post_market_order(self.symbol_b,quantity=-pos_b.qty)
            self.spread_position = None
            logger.warning(f"Stop loss hit: z-score {z_score}")

        elif self.spread_position == "long" and abs(z_score) < self.exit_z:
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
            # Order size limitation
            half_bp = self.portfolio.buying_power() * 0.1 / 2

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







          
            f.write(json.dumps(record) + "\n") mmmm
