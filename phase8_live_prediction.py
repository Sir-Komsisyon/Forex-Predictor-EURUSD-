"""
Phase 8 — Live Prediction Loop
Every 5 minutes: pulls the latest closed candle from MT5, computes the same
features live, predicts direction + confidence using the saved model, and
logs the result to the `predictions` table. Also backfills `actual_direction`
for predictions made one bar ago, so you can track live accuracy over time.

Run this script and leave it running (e.g. in a terminal) during market hours.
"""

import time
import joblib
import pandas as pd
import pandas_ta as ta
import MetaTrader5 as mt5
from datetime import datetime
from db_config import get_engine
from phase4_feature_engineering import compute_features, get_session
from phase6_train_model import FEATURE_COLUMNS

SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
TIMEFRAME_LABEL = "5min"
POLL_SECONDS = 60          # check every minute whether a new 5-min bar has closed
MODEL_VERSION = "xgb_v1"
LOOKBACK_FOR_FEATURES = 100  # bars of history needed to compute rolling indicators


def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def load_model():
    model = joblib.load("xgb_model.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return model, metadata["inverse_map"]


def get_latest_features() -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, LOOKBACK_FOR_FEATURES)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "ts"})
    df = df.rename(columns={"tick_volume": "tick_volume"})

    featured = compute_features(df)
    return featured


def insert_or_get_candle(engine, row) -> int:
    """Ensures the candle exists in the candles table, returns its id."""
    with engine.connect() as conn:
        existing = pd.read_sql(
            f"SELECT id FROM candles WHERE timeframe = '{TIMEFRAME_LABEL}' AND ts = '{row['ts']}'",
            conn,
        )
    if not existing.empty:
        return int(existing.iloc[0]["id"])

    new_row = pd.DataFrame([{
        "timeframe": TIMEFRAME_LABEL, "ts": row["ts"], "open": row["open"],
        "high": row["high"], "low": row["low"], "close": row["close"],
        "tick_volume": row["tick_volume"],
    }])
    new_row.to_sql("candles", engine, if_exists="append", index=False)

    with engine.connect() as conn:
        result = pd.read_sql(
            f"SELECT id FROM candles WHERE timeframe = '{TIMEFRAME_LABEL}' AND ts = '{row['ts']}'",
            conn,
        )
    return int(result.iloc[0]["id"])


def predict_and_log(engine, model, inverse_map):
    featured = get_latest_features()
    latest = featured.iloc[-2]  # second-to-last = the most recently CLOSED bar (last row may be forming)

    if latest[FEATURE_COLUMNS].isna().any():
        print("Not enough warmed-up history yet for indicators, skipping this cycle.")
        return

    candle_id = insert_or_get_candle(engine, latest)

    X = latest[FEATURE_COLUMNS].to_frame().T
    probs = model.predict_proba(X)[0]
    pred_mapped = probs.argmax()
    confidence = probs.max()
    predicted_direction = inverse_map[pred_mapped]

    with engine.connect() as conn:
        already_logged = pd.read_sql(
            f"SELECT id FROM predictions WHERE candle_id = {candle_id} "
            f"AND model_version = '{MODEL_VERSION}'", conn
        )
    if not already_logged.empty:
        print(f"Prediction already logged for candle_id {candle_id}, skipping.")
        return

    row = pd.DataFrame([{
        "candle_id": candle_id,
        "model_version": MODEL_VERSION,
        "predicted_direction": int(predicted_direction),
        "confidence": float(confidence),
        "actual_direction": None,
        "created_at": datetime.utcnow(),
    }])
    row.to_sql("predictions", engine, if_exists="append", index=False)
    print(f"[{datetime.utcnow()}] candle_id={candle_id} predicted={predicted_direction} "
          f"confidence={confidence:.3f}")


def backfill_actuals(engine):
    """
    For predictions where we now know the next bar's close, compute and store
    the actual direction so accuracy can be tracked over time.
    """
    query = """
        SELECT p.id AS pred_id, p.candle_id, c.ts, c.close
        FROM predictions p
        JOIN candles c ON c.id = p.candle_id
        WHERE p.actual_direction IS NULL
    """
    pending = pd.read_sql(query, engine)
    if pending.empty:
        return

    with engine.connect() as conn:
        all_candles = pd.read_sql(
            f"SELECT id, ts, close FROM candles WHERE timeframe = '{TIMEFRAME_LABEL}' ORDER BY ts ASC",
            conn,
        )

    all_candles = all_candles.reset_index(drop=True)
    ts_to_idx = {row["ts"]: i for i, row in all_candles.iterrows()}

    for _, pred in pending.iterrows():
        idx = ts_to_idx.get(pred["ts"])
        if idx is None or idx + 1 >= len(all_candles):
            continue  # next bar hasn't happened yet

        current_close = float(pred["close"])
        next_close = float(all_candles.iloc[idx + 1]["close"])
        pct = (next_close - current_close) / current_close * 100

        actual = 1 if pct > 0.003 else (-1 if pct < -0.003 else 0)

        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"UPDATE predictions SET actual_direction = {actual} WHERE id = {pred['pred_id']}"
            )

    print(f"Backfilled actuals for up to {len(pending)} pending predictions.")


def main():
    connect_mt5()
    model, inverse_map = load_model()
    engine = get_engine()

    print("Starting live prediction loop. Press Ctrl+C to stop.")
    try:
        while True:
            predict_and_log(engine, model, inverse_map)
            backfill_actuals(engine)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("Stopping live loop.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
