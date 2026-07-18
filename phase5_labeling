"""
Phase 5 — Labeling Script
Computes the next-bar % change and direction label (1 / -1 / 0) for each
candle, and writes to the `labels` table.

Horizon used here: next 1 bar (t+1). Change HORIZON_BARS to label further out.
"""

import pandas as pd
from db_config import get_engine

TIMEFRAME_LABEL = "5min"
HORIZON_BARS = 1
HORIZON_NAME = "next_1"

# Dead-zone threshold in percent. Moves smaller than this are labeled "flat" (0)
# instead of forcing every tiny wiggle into up/down noise.
DEADZONE_PCT = 0.003


def load_candles(engine) -> pd.DataFrame:
    query = f"""
        SELECT id AS candle_id, ts, close
        FROM candles
        WHERE timeframe = '{TIMEFRAME_LABEL}'
        ORDER BY ts ASC
    """
    df = pd.read_sql(query, engine)
    df["close"] = df["close"].astype(float)
    return df


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_close"] = df["close"].shift(-HORIZON_BARS)
    df["pct_change"] = (df["future_close"] - df["close"]) / df["close"] * 100

    def classify(pct):
        if pd.isna(pct):
            return None
        if pct > DEADZONE_PCT:
            return 1
        elif pct < -DEADZONE_PCT:
            return -1
        else:
            return 0

    df["direction"] = df["pct_change"].apply(classify)
    df["horizon"] = HORIZON_NAME
    return df.dropna(subset=["direction"])


def write_labels(df: pd.DataFrame, engine):
    cols = ["candle_id", "direction", "pct_change", "horizon"]
    out = df[cols].copy()
    out["direction"] = out["direction"].astype(int)

    with engine.connect() as conn:
        existing = pd.read_sql(
            f"SELECT candle_id FROM labels WHERE horizon = '{HORIZON_NAME}'", conn
        )
    existing_ids = set(existing["candle_id"]) if not existing.empty else set()
    out = out[~out["candle_id"].isin(existing_ids)]

    if out.empty:
        print("No new label rows to insert.")
        return

    out.to_sql("labels", engine, if_exists="append", index=False)
    print(f"Inserted {len(out)} rows into labels table.")


def main():
    engine = get_engine()
    candles = load_candles(engine)
    print(f"Loaded {len(candles)} candles.")

    labeled = compute_labels(candles)
    print("Label distribution:")
    print(labeled["direction"].value_counts())

    write_labels(labeled, engine)


if __name__ == "__main__":
    main()
