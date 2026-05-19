import pandas as pd
import matplotlib.pyplot as plt


def compute_zscore(series):
    return (series - series.mean()) / series.std()


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
