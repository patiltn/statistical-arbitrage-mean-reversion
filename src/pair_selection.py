import pandas as pd


def find_top_correlated_pairs(price_data, top_n=5):
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()

    pairs = []

    tickers = corr_matrix.columns

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            pairs.append((tickers[i], tickers[j], corr_matrix.iloc[i, j]))

    pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    return pairs[:top_n]


if __name__ == "__main__":
    prices = pd.read_csv("data/stock_prices.csv", index_col=0, parse_dates=True)

    top_pairs = find_top_correlated_pairs(prices)

    print("Top correlated pairs:")
    for p1, p2, corr in top_pairs:
        print(f"{p1} - {p2}: correlation = {corr:.4f}")
