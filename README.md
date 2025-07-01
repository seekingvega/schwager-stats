# Schwager Stats
the unofficial library offering the Market Wizard's Trading Performance Evaluation Metrics.

See [this blog post](https://seekingvega.github.io/sv-journal/notebooks/EquityCurve.html) for a more detailed explanation.

## Installation
This package is still in early alpha please install via:
```
pip install git+https://github.com/seekingvega/schwager-stats.git@main
```

## Quick Start
The library provides key metrics including:
- Annualized Return
- Sortino Ratio (risk-adjusted performance)
- Maximum Drawdown (worst peak-to-trough decline)
- Gain-to-Pain Ratio (sum of positive returns divided by absolute sum of negative returns)

```python
import pandas as pd
import numpy as np
from schwager_stats import calculate_performance_metrics

# Create a sample dataframe with closing prices
dates = pd.date_range(start='2020-01-01', end='2022-01-01', freq='B')
close_prices = 100 * (1 + np.random.normal(0.0005, 0.01, len(dates)).cumsum())
df = pd.DataFrame({
    'Close': close_prices,
}, index=dates)

# Calculate performance metrics
metrics = calculate_performance_metrics(
    equity_curve=df,
    risk_free_rate=0.02,  # 2% annual risk-free rate
    trading_days=252,
    price_col='Close'
)

print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# For multiple securities
# Create a multi-index dataframe with multiple tickers
import yfinance as yf
tickers = ['SPY', 'QQQ', 'AAPL']
start_date = '2022-01-01'
end_date = '2023-01-01'

# Download data
data = yf.download(tickers, start=start_date, end=end_date)

# Compute metrics for all tickers
from schwager_stats import compute_performance_metrics
performance_df = compute_performance_metrics(
    df_stocks=data,
    risk_free_rate=0.02,
    trading_days=252,
    price_col='Close'
)

print("\nComparison of multiple securities:")
print(performance_df)
```
