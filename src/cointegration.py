import pandas as pd
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(prices, pvalue_threshold=0.05):
    tickers = prices.columns
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            stock1 = tickers[i]
            stock2 = tickers[j]

            score, pvalue, _ = coint(prices[stock1], prices[stock2])

            results.append((stock1, stock2, pvalue))

    results = sorted(results, key=lambda x: x[2])

    return results


if __name__ == "__main__":
    prices = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)
    prices = prices.dropna()

    pairs = find_cointegrated_pairs(prices)

    print("Cointegration test results:")
    for stock1, stock2, pvalue in pairs:
        print(f"{stock1} - {stock2}: p-value = {pvalue:.4f}")
