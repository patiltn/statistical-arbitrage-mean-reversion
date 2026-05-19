import yfinance as yf
import pandas as pd


def download_stock_data(tickers, start="2018-01-01", end="2026-01-01"):
    data = yf.download(tickers, start=start, end=end)["Close"]
    return data


if __name__ == "__main__":
    tickers = ["MSFT", "AAPL", "GOOG", "AMZN", "META"]

    data = download_stock_data(tickers)

    print(data.head())

    data.to_csv("data/stock_prices.csv")
