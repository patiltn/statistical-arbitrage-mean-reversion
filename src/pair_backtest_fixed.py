import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations

# Trailing window for the hedge ratio.  One year of daily data: long enough
# for a stable OLS slope, short enough to track a relationship that drifts.
BETA_WINDOW = 252
ZSCORE_WINDOW = 30
COST_BPS = 5.0


def compute_zscore(series, window=ZSCORE_WINDOW):
    """Rolling z-score of the spread.

    Rolling, not full-sample: a full-sample mean and standard deviation would
    tell the strategy in 2016 what the spread does in 2025.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def rolling_hedge_ratio(y, x, window=BETA_WINDOW):
    """Trailing OLS slope of y on x, lagged one day.

    This is the fix.  The original version fitted ONE regression over the
    entire sample and used that slope for every historical trade, so each
    position was sized using a relationship estimated partly from the future.
    cov/var over a trailing window is the same estimator applied honestly, and
    `.shift(1)` guarantees the ratio used on day t was computable on day t-1.
    """
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return (cov / var).shift(1)


def full_sample_hedge_ratio(y, x):
    """The ORIGINAL, leaky estimator. Kept so the bias can be measured, not used.

    Retained deliberately: `compare_hedge_ratios` below quantifies what the
    leak was worth, and a claim about look-ahead bias is worth more with the
    number attached than as an assertion.
    """
    model = sm.OLS(y, sm.add_constant(x)).fit()
    return model.params.iloc[1]


def compute_signals(zscore, entry=2.0, exit=0.5):
    """Position that persists between entry and exit.

    Holding through the band matters: the position is open precisely while
    the spread stays wide, so losing days cluster inside one position rather
    than arriving independently.
    """
    signals = pd.Series(index=zscore.index, data=0.0)
    position = 0

    for t in range(len(zscore)):
        z = zscore.iloc[t]
        if pd.isna(z):
            signals.iloc[t] = position
            continue
        if position == 0:
            if z > entry:
                position = -1
            elif z < -entry:
                position = 1
        else:
            if abs(z) < exit:
                position = 0
        signals.iloc[t] = position

    return signals


def compute_sharpe(strategy_returns):
    std = strategy_returns.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (strategy_returns.mean() / std) * np.sqrt(252)


def backtest_pair(prices, stock1, stock2, rolling=True, cost_bps=COST_BPS):
    """Backtest one pair.

    `rolling=True` uses the trailing hedge ratio (correct).
    `rolling=False` reproduces the original full-sample estimator, for
    comparison only.

    Costs are charged on turnover at `cost_bps` of notional per trade.  The
    original backtest charged nothing, which flatters any strategy that
    trades often.
    """
    y = prices[stock1]
    x = prices[stock2]

    try:
        if rolling:
            beta = rolling_hedge_ratio(y, x)
        else:
            beta = pd.Series(full_sample_hedge_ratio(y, x), index=y.index)

        spread = y - beta * x
        zscore = compute_zscore(spread)
        signals = compute_signals(zscore)

        pnl = signals.shift(1) * (y.diff() - beta.shift(1) * x.diff())

        # Absolute value on both legs: a short leg still consumes notional,
        # and a negative beta would otherwise shrink the denominator.
        notional = y.shift(1).abs() + beta.shift(1).abs() * x.shift(1).abs()

        strategy_returns = pnl / notional
        strategy_returns -= (cost_bps / 1e4) * signals.diff().abs().fillna(0)
        strategy_returns = (strategy_returns
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0))

        sharpe = compute_sharpe(strategy_returns)
        total_return = ((1 + strategy_returns).cumprod().iloc[-1] - 1) * 100
        return sharpe, total_return

    except Exception:
        return np.nan, np.nan


def compare_hedge_ratios(prices, pairs=None):
    """Quantify the look-ahead bias: same strategy, both estimators."""
    pairs = pairs or list(combinations(prices.columns, 2))
    rows = []
    for a, b in pairs:
        leaky = backtest_pair(prices, a, b, rolling=False, cost_bps=0.0)
        honest = backtest_pair(prices, a, b, rolling=True)
        rows.append([f"{a}-{b}", leaky[0], leaky[1], honest[0], honest[1]])
    return pd.DataFrame(rows, columns=[
        "Pair", "Sharpe (full-sample beta)", "Return % (full-sample beta)",
        "Sharpe (rolling beta)", "Return % (rolling beta)"])


if __name__ == "__main__":
    prices = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)

    print("=== corrected backtest: rolling hedge ratio, 5bp costs ===")
    results = []
    for stock1, stock2 in combinations(prices.columns, 2):
        sharpe, total_return = backtest_pair(prices, stock1, stock2)
        results.append([stock1, stock2, sharpe, total_return])

    results_df = pd.DataFrame(
        results, columns=["Stock 1", "Stock 2", "Sharpe Ratio", "Total Return (%)"]
    ).sort_values(by="Sharpe Ratio", ascending=False)
    print(results_df.to_string(index=False))
    results_df.to_csv("data/pair_results.csv", index=False)

    print("\n=== look-ahead bias: what the full-sample hedge ratio was worth ===")
    print(compare_hedge_ratios(prices).to_string(index=False))
