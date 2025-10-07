# Configuration settings for the factor analysis system

# Baostock login credentials
BAOSTOCK_USER = ""
BAOSTOCK_PASSWORD = ""

# Data settings
START_DATE = "2018-01-01"  # 缩短时间段以加快演示
END_DATE = "2023-12-31"
UNIVERSE = ["sh.000300"]  # CSI 300 as default universe

# Analysis settings
RETURN_PERIOD = 21  # 1-month forward returns
QUANTILES = 5  # Number of quantile groups for analysis
MIN_PERIODS = 12  # Minimum periods required for factor calculation

# Visualization settings
PLOT_STYLE = "default"  # 使用默认样式而不是seaborn
FIG_SIZE = (12, 8)