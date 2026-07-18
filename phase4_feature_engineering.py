"""
Phase 4 — Feature Engineering Script
Reads raw candles from MySQL, computes technical indicators + time-context
features + lagged returns, and writes results to the `features` table.
"""

import pandas as pd
import pandas_ta as ta
from db_config import get_engine

TIMEFRAME_LABEL = "5min"


def load_candles(engine) -> pd.DataFrame:
    query = f"""
        SELECT id AS candle_id, ts, open, high, low, close, tick_volume
        FROM candles
        WHERE timeframe = '{TIMEFRAME_LABEL}'
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, engine)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df


def get_session(hour: int) -> str:
    # Rough UTC session windows — adjust if your MT5 server time differs from UTC.
    if 0 <= hour < 7:
        return "asian"
    elif 7 <= hour < 12:
        return "london"
    elif 12 <= hour < 16:
        return "overlap"
    elif 16 <= hour < 21:
        return "ny"
    else:
        return "asian"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["sma_10"] = ta.sma(df["close"], length=10)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]

    bbands = ta.bbands(df["close"], length=20)
    df["bb_upper"] = bbands["BBU_20_2.0"]
    df["bb_lower"] = bbands["BBL_20_2.0"]

    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Rolling volatility = std dev of % returns over last 20 bars
    df["returns"] = df["close"].pct_change()
    df["rolling_volatility"] = df["returns"].rolling(window=20).std()

    # Lagged returns (autocorrelation signal) — kept in df for training,
    # not all of these have dedicated SQL columns; core ones are stored below.
    df["lag_return_1"] = df["returns"].shift(1)
    df["lag_return_2"] = df["returns"].shift(2)
    df["lag_return_3"] = df["returns"].shift(3)

    # Time context
    df["hour_of_day"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.dayofweek  # Monday=0
    df["session"] = df["hour_of_day"].apply(get_session)

    return df


def write_features(df: pd.DataFrame, engine):
    cols = [
        "candle_id", "sma_10", "sma_50", "ema_10", "rsi_14",
        "macd", "macd_signal", "bb_upper", "bb_lower", "atr_14",
        "rolling_volatility", "hour_of_day", "day_of_week", "session"
    ]
    out = df[cols].dropna(subset=["sma_50"])  # drop early rows where indicators aren't warmed up yet

    with engine.connect() as conn:
        existing = pd.read_sql("SELECT candle_id FROM features", conn)
    existing_ids = set(existing["candle_id"]) if not existing.empty else set()

    out = out[~out["candle_id"].isin(existing_ids)]

    if out.empty:
        print("No new feature rows to insert.")
        return

    out.to_sql("features", engine, if_exists="append", index=False)
    print(f"Inserted {len(out)} rows into features table.")


def main():
    engine = get_engine()
    candles = load_candles(engine)
    print(f"Loaded {len(candles)} candles.")

    featured = compute_features(candles)
    write_features(featured, engine)

    # Save the full featured dataframe (including lag_return columns not in SQL)
    # to a local pickle so Phase 6 training can use the richer feature set directly
    # without re-querying/re-computing everything.
    featured.to_pickle("featured_dataset.pkl")
    print("Saved featured_dataset.pkl for use in training.")


if __name__ == "__main__":
    main()
