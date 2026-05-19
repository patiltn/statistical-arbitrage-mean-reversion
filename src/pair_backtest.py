import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import combinations


def compute_zscore(series):
    rolling_mean = series.rolling(30).mean()
    rolling_std = series.rolling(30).std()

    return (series - rolling_mean) / rolling_std


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

        signals = pd.Series(
            index=zscore.index,
            data=0
        )

        signals[zscore > 2] = -1
        signals[zscore < -2] = 1
        signals[abs(zscore) < 0.5] = 0

        spread_returns = spread.diff()

        strategy_returns = (
            signals.shift(1)
            * spread_returns
        )

        sharpe = compute_sharpe(
            strategy_returns
        )

        total_return = (
            (
                1
                + strategy_returns.fillna(0)
                / 100
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
