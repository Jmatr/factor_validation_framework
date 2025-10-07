import pandas as pd
import numpy as np
from factors.factor_base import Factor


class CompositeValueFactor(Factor):
    def __init__(self):
        super().__init__("VALUE_COMPOSITE", "Composite value factor using multiple metrics")

    def calculate(self, data: dict) -> pd.DataFrame:
        value_scores = []

        # P/E ratio (inverse)
        if 'peTTM' in data:
            pe_score = 1 / data['peTTM']
            pe_score = pe_score.replace([np.inf, -np.inf], np.nan)
            value_scores.append(pe_score)

        # P/B ratio (inverse)
        if 'pbMRQ' in data:
            pb_score = 1 / data['pbMRQ']
            pb_score = pb_score.replace([np.inf, -np.inf], np.nan)
            value_scores.append(pb_score)

        # P/S ratio (inverse)
        if 'psTTM' in data:
            ps_score = 1 / data['psTTM']
            ps_score = ps_score.replace([np.inf, -np.inf], np.nan)
            value_scores.append(ps_score)

        if value_scores:
            # 修复：确保返回的是DataFrame而不是Series
            combined_value = pd.concat(value_scores, axis=1).mean(axis=1)
            # 转换为DataFrame格式
            combined_df = pd.DataFrame(combined_value)
            combined_df = combined_df.pivot_table(index=combined_df.index, columns=combined_df.columns[0] if len(
                combined_df.columns) > 1 else 'value')
            return self.winsorize_factor(combined_df)
        else:
            raise ValueError("No value data available")


class QualityROEFactor(Factor):
    def __init__(self):
        super().__init__("QUALITY_ROE", "Quality factor based on ROE")

    def calculate(self, data: dict) -> pd.DataFrame:
        # For demonstration, we'll use PB and PE to estimate ROE
        # ROE ≈ (1/PE) / (1/PB) = PB/PE
        self.validate_data(data, ['peTTM', 'pbMRQ'])
        roe = data['pbMRQ'] / data['peTTM']
        roe = roe.replace([np.inf, -np.inf], np.nan)
        return self.winsorize_factor(roe)


class GrowthFactor(Factor):
    def __init__(self, lookback_period: int = 252):
        super().__init__(f"GROWTH_{lookback_period}",
                         f"Growth factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.parameters = {'lookback_period': lookback_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        # Use price growth as proxy for earnings growth
        growth = data['close'].pct_change(self.lookback_period)
        return self.winsorize_factor(growth)


class RSIFactor(Factor):
    def __init__(self, lookback_period: int = 14):
        super().__init__(f"RSI_{lookback_period}",
                         f"RSI technical factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.parameters = {'lookback_period': lookback_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        close = data['close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.lookback_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.lookback_period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return self.winsorize_factor(rsi)


class MACDFactor(Factor):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__("MACD", "MACD technical factor")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        }

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        close = data['close']

        ema_fast = close.ewm(span=self.fast_period).mean()
        ema_slow = close.ewm(span=self.slow_period).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=self.signal_period).mean()
        macd_histogram = macd - macd_signal

        return self.winsorize_factor(macd_histogram)


class BollingerBandsFactor(Factor):
    def __init__(self, lookback_period: int = 20, std_dev: int = 2):
        super().__init__(f"BOLL_{lookback_period}",
                         f"Bollinger Bands factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.std_dev = std_dev
        self.parameters = {'lookback_period': lookback_period, 'std_dev': std_dev}

    def calculate(self, data: dict) -> pd.DataFrame:
        self.validate_data(data, ['close'])
        close = data['close']

        sma = close.rolling(window=self.lookback_period).mean()
        std = close.rolling(window=self.lookback_period).std()

        upper_band = sma + (std * self.std_dev)
        lower_band = sma - (std * self.std_dev)

        # Bollinger Band position
        bb_position = (close - lower_band) / (upper_band - lower_band)

        return self.winsorize_factor(bb_position)


class ATRFactor(Factor):
    def __init__(self, lookback_period: int = 14):
        super().__init__(f"ATR_{lookback_period}",
                         f"Average True Range factor with {lookback_period} days lookback")
        self.lookback_period = lookback_period
        self.parameters = {'lookback_period': lookback_period}

    def calculate(self, data: dict) -> pd.DataFrame:
        # 修复：检查所有必需的数据字段
        required_fields = ['high', 'low', 'close']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            print(f"  ATR factor skipped: missing fields {missing_fields}")
            return pd.DataFrame()  # 返回空的DataFrame而不是None

        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.lookback_period).mean()

        # Normalize by price
        atr_normalized = atr / close

        return self.winsorize_factor(atr_normalized)


class CompositeQualityFactor(Factor):
    def __init__(self):
        super().__init__("QUALITY_COMPOSITE", "Composite quality factor")

    def calculate(self, data: dict) -> pd.DataFrame:
        quality_scores = []

        # Profitability (using ROE proxy)
        if 'peTTM' in data and 'pbMRQ' in data:
            roe = data['pbMRQ'] / data['peTTM']
            roe = roe.replace([np.inf, -np.inf], np.nan)
            quality_scores.append(roe)

        # Stability (low volatility)
        if 'close' in data:
            volatility = data['close'].pct_change().rolling(window=63).std()
            low_vol_score = -volatility  # Lower volatility is better
            quality_scores.append(low_vol_score)

        # Liquidity (turnover)
        if 'turn' in data:
            liquidity = data['turn']
            quality_scores.append(liquidity)

        if quality_scores:
            # 修复：确保返回的是DataFrame而不是Series
            combined_quality = pd.concat(quality_scores, axis=1).mean(axis=1)
            # 转换为DataFrame格式
            combined_df = pd.DataFrame(combined_quality)
            combined_df = combined_df.pivot_table(index=combined_df.index, columns=combined_df.columns[0] if len(
                combined_df.columns) > 1 else 'value')
            return self.winsorize_factor(combined_df)
        else:
            raise ValueError("No quality data available")