import yfinance as yf
import pandas as pd

def download_data(ticker="AAPL", start="2019-01-01", end="2024-01-01"):
    data = yf.download(ticker, start=start, end=end)
    data.to_csv("raw_data.csv")
    print("Data saved to raw_data.csv")

if __name__ == "__main__":
    download_data()