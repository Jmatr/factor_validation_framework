# Configuration settings for the factor analysis system

# Baostock login credentials
BAOSTOCK_USER = ""
BAOSTOCK_PASSWORD = ""

# Data settings
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
UNIVERSE = ["sh.000300"]  # CSI 300 as default universe

# Analysis settings
RETURN_PERIOD = 21  # 1-month forward returns
QUANTILES = 5  # Number of quantile groups for analysis
MIN_PERIODS = 12  # Minimum periods required for factor calculation
MIN_STOCKS_PER_QUANTILE = 3  # Minimum stocks per quantile

# Backtest settings
INITIAL_CAPITAL = 1000000
TRANSACTION_COST = 0.001  # 0.1% transaction cost

# Visualization settings
PLOT_STYLE = "default"
FIG_SIZE = (12, 8)
COLOR_PALETTE = "viridis"

# Risk analysis
BENCHMARK = "sh.000300"  # CSI 300 as benchmark
RISK_FREE_RATE = 0.03  # 3% annual risk-free rate

# Performance thresholds
SIGNIFICANCE_LEVEL_1 = 2.58  # 1% significance
SIGNIFICANCE_LEVEL_5 = 1.96  # 5% significance
SIGNIFICANCE_LEVEL_10 = 1.65  # 10% significance