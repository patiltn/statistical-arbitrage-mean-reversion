# Statistical Arbitrage in Equity Markets

A quantitative finance project implementing and evaluating a mean-reversion statistical arbitrage strategy on US equities.

## Overview

This project explores whether correlated equity pairs exhibit statistically exploitable mean-reverting behaviour.

The workflow includes:

- Historical market data collection
- Correlation analysis
- Cointegration testing
- Hedge ratio estimation via regression
- Z-score signal generation
- Strategy backtesting
- Risk-adjusted performance evaluation

## Methodology

### 1. Data Collection

Historical stock prices were downloaded using Yahoo Finance.

Assets analysed:

- AAPL
- AMZN
- GOOG
- META
- MSFT

### 2. Pair Selection

Pairs were initially screened using correlation analysis.

Cointegration testing was then applied to identify stable long-term statistical relationships.

### 3. Statistical Arbitrage Strategy

For each pair:

- Hedge ratio estimated using OLS regression
- Spread constructed:

spread = y - βx

- Rolling z-score computed
- Trading signals generated:

| Condition | Action |
|------------|--------|
| z-score > 2 | Short spread |
| z-score < -2 | Long spread |
| abs(z-score) < 0.5 | Exit |

### 4. Backtesting Metrics

Performance evaluated using:

- Sharpe Ratio
- Total Return
- Annual Volatility
- Maximum Drawdown

## Results

| Pair | Sharpe Ratio | Total Return |
|------|--------------|--------------|
| AAPL-GOOG | 0.56 | 82.67% |
| AAPL-AMZN | 0.44 | 77.31% |
| AAPL-META | 0.39 | 44.92% |

Best-performing pair:

**AAPL – GOOG**

## Repository Structure

```text
data/
figures/
notebooks/
src/