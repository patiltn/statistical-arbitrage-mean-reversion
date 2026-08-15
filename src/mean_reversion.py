import pandas as pd
import matplotlib.pyplot as plt


def compute_zscore(series, window=30):
    """
    Rolling z-score, consistent with backtest.py. The original version
    used the full-sample mean and std, which means the z-score at an
    early date was influenced by data from years in the future -- not
    something a real trader would have known at the time.
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    return (series - rolling_mean) / rolling_std


if __name__ == "__main__":
    prices = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)

    stock1 = "GOOG"
    stock2 = "MSFT"

    spread = prices[stock1] - prices[stock2]
    zscore = compute_zscore(spread)

    plt.figure(figsize=(10, 5))
    plt.plot(zscore)
    plt.axhline(2, linestyle="--")
    plt.axhline(-2, linestyle="--")
    plt.axhline(0)

    plt.title(f"Z-score Spread: {stock1} vs {stock2}")
    plt.xlabel("Date")
    plt.ylabel("Z-score")
    plt.tight_layout()

    plt.savefig("figures/zscore_spread_GOOG_MSFT.png")
    plt.show()
