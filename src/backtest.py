import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def compute_zscore(series):
    rolling_mean = series.rolling(30).mean()
    rolling_std = series.rolling(30).std()

    return (series - rolling_mean) / rolling_std


def compute_max_drawdown(portfolio):
    running_max = portfolio.cummax()
    drawdown = (
        portfolio - running_max
    ) / running_max

    return drawdown.min()


if __name__ == "__main__":
    prices = pd.read_csv(
        "data/stock_prices.csv",
        index_col=0,
        parse_dates=True
    )

    stock1 = "AMZN"
    stock2 = "META"

    y = prices[stock1]
    x = prices[stock2]

    model = sm.OLS(y, sm.add_constant(x)).fit()

    hedge_ratio = model.params.iloc[1]

    spread = y - hedge_ratio * x

    zscore = compute_zscore(spread)

    signals = pd.Series(index=zscore.index, data=0)

    signals[zscore > 2] = -1
    signals[zscore < -2] = 1
    signals[abs(zscore) < 0.5] = 0

    spread_returns = spread.diff()

    strategy_returns = (
        signals.shift(1)
        * spread_returns
    )

    cumulative_returns = (
        1 + strategy_returns.fillna(0) / 100
    ).cumprod()

    sharpe_ratio = (
        strategy_returns.mean()
        / strategy_returns.std()
    ) * np.sqrt(252)

    total_return = (
        cumulative_returns.iloc[-1] - 1
    ) * 100

    volatility = (
        strategy_returns.std()
        * np.sqrt(252)
    )

    max_drawdown = (
        compute_max_drawdown(
            cumulative_returns
        ) * 100
    )

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns)

    plt.title("Statistical Arbitrage Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")

    plt.tight_layout()

    plt.savefig(
        "figures/backtest_returns.png"
    )

    plt.show()

    print(f"Hedge Ratio (β): {hedge_ratio:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annual Volatility: {volatility:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
