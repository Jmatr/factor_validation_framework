### 5. 因子库 (factors/factor_library.py)

import pandas as pd
import numpy as np
from factors.factor_base import Factor


class MomentumFactor(Factor):
    def __init__(self, lookback_period=21):
        super().__init__(f"MOM_{lookback_period}",
                         f"Momentum factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period

    def calculate(self, data):
        close_prices = data['close']
        momentum = close_prices.pct_change(self.lookback_period)
        return momentum


class ValueFactor(Factor):
    def __init__(self):
        super().__init__("VALUE", "Value factor based on P/E ratio")

    def calculate(self, data):
        if 'peTTM' in data:
            # Use inverse of P/E (E/P) as value measure
            value = 1 / data['peTTM']
            # Handle infinite values
            value = value.replace([np.inf, -np.inf], np.nan)
            return value
        else:
            raise ValueError("P/E data not available")


class SizeFactor(Factor):
    def __init__(self):
        super().__init__("SIZE", "Size factor based on market capitalization")

    def calculate(self, data):
        if 'close' in data and 'volume' in data:
            # Simple proxy for market cap: price * volume
            market_cap = data['close'] * data['volume']
            # Use log market cap
            size = np.log(market_cap)
            return size
        else:
            raise ValueError("Close price or volume data not available")


class VolatilityFactor(Factor):
    def __init__(self, lookback_period=21):
        super().__init__(f"VOL_{lookback_period}",
                         f"Volatility factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period

    def calculate(self, data):
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.lookback_period).std()
        return volatility


class QualityFactor(Factor):
    def __init__(self):
        super().__init__("QUALITY", "Quality factor based on profitability")

    def calculate(self, data):
        # Simple quality proxy: high turnover might indicate good quality
        if 'turn' in data:
            quality = data['turn']
            return quality
        else:
            raise ValueError("Turnover data not available")


# Factor factory class
class FactorFactory:
    @staticmethod
    def create_factor(factor_type, **kwargs):
        if factor_type == "momentum":
            return MomentumFactor(**kwargs)
        elif factor_type == "value":
            return ValueFactor()
        elif factor_type == "size":
            return SizeFactor()
        elif factor_type == "volatility":
            return VolatilityFactor(**kwargs)
        elif factor_type == "quality":
            return QualityFactor()
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")