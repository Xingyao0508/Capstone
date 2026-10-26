"""Shared paths, constants and raw loaders for the Walmart / EverTrail pipeline."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
TABLES, FIGS, MODELS = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "outputs" / "models"
for _p in (PROC, TABLES, FIGS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

OUTCOME_DAYS = 14
OFFERS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
HIST_CAMPAIGNS = [f"CP0{i}" for i in range(1, 8)]

TS_COLS = {
    "campaigns": ["send_ts"], "historical_assignments": ["send_ts"], "web_sessions": ["session_ts"],
    "email_events": ["event_ts"], "orders": ["order_ts"], "returns": ["return_ts"],
    "policy_scoring_population": ["policy_as_of_ts"], "customers": ["signup_date"],
}
TABLE_NAMES = ["campaigns", "customers", "historical_assignments", "web_sessions", "email_events", "orders",
               "order_items", "products", "returns", "offer_catalog", "policy_scoring_population"]


def load_raw(name):
    return pd.read_csv(RAW / f"{name}.csv", parse_dates=TS_COLS.get(name, []), date_format="%Y-%m-%d %H:%M:%S"
                       if name != "customers" else "%Y-%m-%d")


def load_all_raw():
    return {n: load_raw(n) for n in TABLE_NAMES}


def save(df, name, folder=PROC, parquet=True):
    df.to_csv(folder / f"{name}.csv", index=False)
    if parquet:
        df.to_parquet(folder / f"{name}.parquet", index=False)


def load_clean(names=TABLE_NAMES):
    """Step 01 clean core tables (data/processed/clean)."""
    return {n: pd.read_parquet(PROC / "clean" / f"{n}.parquet") for n in names}


def load_processed(name):
    p = PROC / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(PROC / f"{name}.csv")
