# Factor configurations and parameters

FACTOR_CONFIGS = {
    # Momentum factors
    "momentum_short": {"lookback_period": 21, "skip_period": 1},
    "momentum_medium": {"lookback_period": 63, "skip_period": 5},
    "momentum_long": {"lookback_period": 252, "skip_period": 21},

    # Value factors
    "value_pe": {},
    "value_pb": {},
    "value_ps": {},
    "value_composite": {},

    # Quality factors
    "quality_gross_profit": {},
    "quality_roe": {},
    "quality_roa": {},
    "quality_composite": {},

    # Growth factors
    "growth_revenue": {"lookback_period": 252},
    "growth_earnings": {"lookback_period": 252},
    "growth_composite": {"lookback_period": 252},

    # Size factors
    "size": {},

    # Volatility factors
    "volatility_short": {"lookback_period": 21},
    "volatility_medium": {"lookback_period": 63},
    "volatility_long": {"lookback_period": 252},

    # Technical factors
    "rsi": {"lookback_period": 14},
    "macd": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    "bollinger_bands": {"lookback_period": 20, "std_dev": 2},
    "atr": {"lookback_period": 14},

    # Reversal factors
    "reversal_short": {"lookback_period": 1},
    "reversal_medium": {"lookback_period": 5},
    "reversal_long": {"lookback_period": 21},
}

# Factor groups for analysis
FACTOR_GROUPS = {
    "momentum": ["momentum_short", "momentum_medium", "momentum_long"],
    "value": ["value_pe", "value_pb", "value_ps", "value_composite"],
    "quality": ["quality_gross_profit", "quality_roe", "quality_roa", "quality_composite"],
    "growth": ["growth_revenue", "growth_earnings", "growth_composite"],
    "technical": ["rsi", "macd", "bollinger_bands", "atr"],
    "volatility": ["volatility_short", "volatility_medium", "volatility_long"],
    "reversal": ["reversal_short", "reversal_medium", "reversal_long"],
    "size": ["size"]
}