import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def compute_zscore(series):
    rolling_mean = series.rolling(30).mean()
    rolling_std = series.rolling(30).std()

    return (series - rolling_mean) / rolling_std


def compute_signals(zscore, entry=2.0, exit=0.5):
    """
    Generates a position signal that persists between entry and exit,
    rather than re-evaluating each day independently. A position, once
    opened, is held until the spread reverts inside the exit band.
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

    stock1 = "AAPL"
    stock2 = "MSFT"

    y = prices[stock1]
    x = prices[stock2]

    model = sm.OLS(y, sm.add_constant(x)).fit()

    hedge_ratio = model.params.iloc[1]

    spread = y - hedge_ratio * x

    zscore = compute_zscore(spread)

    signals = compute_signals(zscore)

    spread_returns = spread.diff()

    # Normalize by position notional (long leg + short leg dollar value)
    # instead of dividing by a flat constant, so the return is a genuine
    # percentage regardless of the pair's price scale.
    notional = y.shift(1).abs() + hedge_ratio * x.shift(1).abs()

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

    cumulative_returns = (1 + strategy_returns).cumprod()

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
    ) * 100

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
        "figures/backtest_returns_AAPL_MSFT.png"
    )

    plt.show()

    print(f"Hedge Ratio (β): {hedge_ratio:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annual Volatility: {volatility:.2f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
