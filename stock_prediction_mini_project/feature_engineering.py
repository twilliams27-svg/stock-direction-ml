import pandas as pd

def add_features():
    df = pd.read_csv("raw_data.csv", index_col=0, parse_dates=True)

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["High"] = pd.to_numeric(df["High"], errors="coerce")
    df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # Simple Features
    df["Return"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["RSI"] = compute_rsi(df["Close"], period=14)

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    df.to_csv("feature_data.csv", index=False)
    print("Features added and saved to feature_data.csv")

def compute_rsi(series, period=14):

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    ma_up = gain.rolling(period).mean()
    ma_down = loss.rolling(period).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

if __name__ == "__main__":
    add_features()