import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations


def compute_zscore(series):
    rolling_mean = series.rolling(30).mean()
    rolling_std = series.rolling(30).std()

    return (series - rolling_mean) / rolling_std


def compute_signals(zscore, entry=2.0, exit=0.5):
    """
    Position signal that persists between entry and exit rather than
    resetting to flat whenever the z-score dips back inside the entry
    band before reaching the actual exit threshold.
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

    return (
        strategy_returns.mean()
        / std
    ) * np.sqrt(252)


def backtest_pair(prices, stock1, stock2):

    y = prices[stock1]
    x = prices[stock2]

    try:
        model = sm.OLS(
            y,
            sm.add_constant(x)
        ).fit()

        hedge_ratio = model.params.iloc[1]

        spread = y - hedge_ratio * x

        zscore = compute_zscore(spread)

        signals = compute_signals(zscore)

        spread_returns = spread.diff()

        # Normalize by position notional instead of a flat constant, so
        # returns are comparable across pairs trading at different price
        # scales.
        notional = (
            y.shift(1).abs()
            + hedge_ratio * x.shift(1).abs()
        )

        strategy_returns = (
            signals.shift(1)
            * spread_returns
            / notional
        )

        strategy_returns = (
            strategy_returns
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        sharpe = compute_sharpe(
            strategy_returns
        )

        total_return = (
            (
                1 + strategy_returns
            ).cumprod().iloc[-1]
            - 1
        ) * 100

        return sharpe, total_return

    except Exception:
        return np.nan, np.nan


if __name__ == "__main__":

    prices = pd.read_csv(
        "data/stock_prices.csv",
        index_col=0,
        parse_dates=True
    )

    tickers = prices.columns

    results = []

    for stock1, stock2 in combinations(
        tickers, 2
    ):

        sharpe, total_return = (
            backtest_pair(
                prices,
                stock1,
                stock2
            )
        )

        results.append(
            [
                stock1,
                stock2,
                sharpe,
                total_return
            ]
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Stock 1",
            "Stock 2",
            "Sharpe Ratio",
            "Total Return (%)"
        ]
    )

    results_df = (
        results_df
        .sort_values(
            by="Sharpe Ratio",
            ascending=False
        )
    )

    print(results_df)

    results_df.to_csv(
        "data/pair_results.csv",
        index=False
    )
