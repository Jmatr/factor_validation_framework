import pandas as pd
import numpy as np
from factors.factor_base import Factor


class MomentumFactor(Factor):
    def __init__(self, lookback_period: int = 21, skip_period: int = 1):
        super().__init__(f"MOM_{lookback_period}",
                         f"Momentum factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.skip_period = skip_period
        self.parameters = {'lookback_period': lookback_period, 'skip_period': skip_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        close_prices = data['close']
        # Skip recent period to avoid short-term reversal
        momentum = (close_prices / close_prices.shift(self.lookback_period + self.skip_period) - 1)
        return self.winsorize_factor(momentum)


class ValueFactor(Factor):
    def __init__(self):
        super().__init__("VALUE_PE", "Value factor based on P/E ratio")

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['peTTM'])
        # Use inverse of P/E (E/P) as value measure
        value = 1 / data['peTTM']
        # Handle infinite values
        value = value.replace([np.inf, -np.inf], np.nan)
        return self.winsorize_factor(value)


class SizeFactor(Factor):
    def __init__(self):
        super().__init__("SIZE", "Size factor based on market capitalization")

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close', 'volume'])
        # Simple proxy for market cap: price * volume
        market_cap = data['close'] * data['volume']
        # Use log market cap
        size = np.log(market_cap)
        return self.winsorize_factor(size)


class VolatilityFactor(Factor):
    def __init__(self, lookback_period: int = 21):
        super().__init__(f"VOL_{lookback_period}",
                         f"Volatility factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.parameters = {'lookback_period': lookback_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.lookback_period).std()
        return self.winsorize_factor(volatility)


class QualityFactor(Factor):
    def __init__(self):
        super().__init__("QUALITY_TURN", "Quality factor based on turnover")

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['turn'])
        # Higher turnover might indicate better quality (liquidity)
        quality = data['turn']
        return self.winsorize_factor(quality)


class ReversalFactor(Factor):
    def __init__(self, lookback_period: int = 1):
        super().__init__(f"REV_{lookback_period}",
                         f"Reversal factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.parameters = {'lookback_period': lookback_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        returns = data['close'].pct_change(self.lookback_period)
        # Negative sign for reversal (past losers become winners)
        reversal = -returns
        return self.winsorize_factor(reversal)