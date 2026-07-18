"""
Phase 3 - Historical Data Ingestion

Downloads EURUSD candles from MetaTrader 5
and stores them inside MySQL.
"""

import MetaTrader5 as mt5
import pandas as pd

from db_config import get_engine


# ---------------- SETTINGS ---------------- #

SYMBOL = "EURUSD"

TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_LABEL = "5min"

NUM_CANDLES = 50000


# ------------------------------------------ #

def connect_mt5():

    print("Connecting to MetaTrader 5...")

    if not mt5.initialize():
        raise RuntimeError(
            f"MT5 initialization failed.\nError: {mt5.last_error()}"
        )

    print("✓ MT5 Connected")

    info = mt5.symbol_info(SYMBOL)

    if info is None:
        raise RuntimeError(
            f"{SYMBOL} not found in MT5."
        )

    if not info.visible:
        mt5.symbol_select(SYMBOL, True)


def fetch_candles(symbol, timeframe, count):

    print(f"Downloading {count} candles...")

    rates = mt5.copy_rates_from_pos(
        symbol,
        timeframe,
        0,
        count
    )

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            "No candles returned from MT5."
        )

    df = pd.DataFrame(rates)

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s"
    )

    df.rename(
        columns={"time": "ts"},
        inplace=True
    )

    df["timeframe"] = TIMEFRAME_LABEL

    df = df[
        [
            "timeframe",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "tick_volume"
        ]
    ]

    return df


def insert_candles(df):

    engine = get_engine()

    print("Checking existing candles...")

    existing = pd.read_sql(
        f"""
        SELECT ts
        FROM candles
        WHERE timeframe='{TIMEFRAME_LABEL}'
        """,
        engine,
    )

    existing_ts = set(existing["ts"].tolist())

    new_rows = df[
        ~df["ts"].isin(existing_ts)
    ]

    if len(new_rows) == 0:
        print("Database already up-to-date.")
        return

    new_rows.to_sql(
        "candles",
        engine,
        if_exists="append",
        index=False,
    )

    print(f"Inserted {len(new_rows)} candles.")


def main():

    print("=" * 50)
    print("EURUSD Historical Data Import")
    print("=" * 50)

    connect_mt5()

    try:

        df = fetch_candles(
            SYMBOL,
            TIMEFRAME,
            NUM_CANDLES
        )

        print(
            f"Downloaded {len(df)} candles\n"
            f"{df['ts'].min()} --> {df['ts'].max()}"
        )

        insert_candles(df)

    finally:

        mt5.shutdown()

        print("MT5 Closed")


if __name__ == "__main__":
    main()
