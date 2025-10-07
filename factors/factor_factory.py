from factors.factor_library import *
from factors.enhanced_factors import *


class FactorFactory:
    @staticmethod
    def create_factor(factor_type: str, **kwargs):
        factor_map = {
            # Basic factors
            "momentum": MomentumFactor,
            "value_pe": ValueFactor,
            "size": SizeFactor,
            "volatility": VolatilityFactor,
            "quality_turn": QualityFactor,
            "reversal": ReversalFactor,

            # Enhanced factors
            "value_composite": CompositeValueFactor,
            "quality_roe": QualityROEFactor,
            "quality_composite": CompositeQualityFactor,
            "growth": GrowthFactor,
            "rsi": RSIFactor,
            "macd": MACDFactor,
            "bollinger_bands": BollingerBandsFactor,
            "atr": ATRFactor,
        }

        if factor_type in factor_map:
            factor_class = factor_map[factor_type]
            return factor_class(**kwargs)
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")


class FactorGroupFactory:
    @staticmethod
    def create_factor_group(group_name: str, configs: dict):
        """Create a group of factors with different parameters"""
        factors = []

        if group_name == "momentum":
            for lookback in [21, 63, 252]:
                factors.append(FactorFactory.create_factor(
                    "momentum", lookback_period=lookback, skip_period=1
                ))

        elif group_name == "value":
            factors.extend([
                FactorFactory.create_factor("value_pe"),
                FactorFactory.create_factor("value_composite"),
            ])

        elif group_name == "quality":
            factors.extend([
                FactorFactory.create_factor("quality_turn"),
                FactorFactory.create_factor("quality_roe"),
                FactorFactory.create_factor("quality_composite"),
            ])

        elif group_name == "technical":
            factors.extend([
                FactorFactory.create_factor("rsi", lookback_period=14),
                FactorFactory.create_factor("macd"),
                FactorFactory.create_factor("bollinger_bands", lookback_period=20),
                FactorFactory.create_factor("atr", lookback_period=14),
            ])

        elif group_name == "volatility":
            for lookback in [21, 63, 252]:
                factors.append(FactorFactory.create_factor(
                    "volatility", lookback_period=lookback
                ))

        elif group_name == "reversal":
            for lookback in [1, 5, 21]:
                factors.append(FactorFactory.create_factor(
                    "reversal", lookback_period=lookback
                ))

        else:
            raise ValueError(f"Unknown factor group: {group_name}")

        return factors