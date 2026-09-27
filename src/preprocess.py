"""
preprocess.py
-------------
Loads trains.csv and maintenance.csv, cleans them, converts types,
and validates the data. Returns clean DataFrames ready for ML and optimization.
"""

import pandas as pd
import numpy as np
import os

PRIORITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _to_minutes(time_str):
    """Convert 'HH:MM' string to total minutes since midnight."""
    try:
        h, m = map(int, str(time_str).strip().split(":"))
        return h * 60 + m
    except Exception:
        return np.nan


def load_trains(path=None):
    """
    Load and clean the trains CSV.
    - Converts arrival/departure to minutes-since-midnight (integers).
    - Fills missing priorities with 'Medium'.
    - Adds numeric priority column.
    - Drops rows with invalid times.
    """
    path = path or os.path.join(DATA_DIR, "trains.csv")
    df = pd.read_csv(path)

    # Fill missing values
    df["train_priority"] = df["train_priority"].fillna("Medium")
    df["section_id"] = df["section_id"].fillna("UNKNOWN")

    # Convert times to minutes
    df["arr_min"] = df["arrival_time"].apply(_to_minutes)
    df["dep_min"] = df["departure_time"].apply(_to_minutes)

    # Handle overnight trains: if departure < arrival, add 1440 (24h)
    mask = df["dep_min"] < df["arr_min"]
    df.loc[mask, "dep_min"] = df.loc[mask, "dep_min"] + 1440

    # Drop rows where time conversion failed
    before = len(df)
    df = df.dropna(subset=["arr_min", "dep_min"])
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} train rows with invalid times.")

    # Numeric priority
    df["priority_num"] = df["train_priority"].map(PRIORITY_MAP).fillna(2)

    df = df.reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} train records.")
    return df


def load_maintenance(path=None):
    """
    Load and clean the maintenance CSV.
    - Converts maintenance_deadline to datetime.
    - Converts boolean columns.
    - Adds numeric priority column.
    - Fills missing values with sensible defaults.
    """
    path = path or os.path.join(DATA_DIR, "maintenance.csv")
    df = pd.read_csv(path)

    # Fill missing values
    df["maintenance_priority"] = df["maintenance_priority"].fillna("Medium")
    df["maintenance_duration_hours"] = df["maintenance_duration_hours"].fillna(2)
    df["asset_condition"] = df["asset_condition"].fillna(3)
    df["worker_available"] = df["worker_available"].fillna(True)
    df["machine_available"] = df["machine_available"].fillna(True)

    # Convert deadline to datetime
    df["maintenance_deadline"] = pd.to_datetime(
        df["maintenance_deadline"], errors="coerce"
    )

    # Convert bool columns (CSV stores them as strings sometimes)
    for col in ["worker_available", "machine_available"]:
        if df[col].dtype == object:
            df[col] = df[col].map(
                {"True": True, "False": False, True: True, False: False}
            ).fillna(True).astype(bool)

    # Numeric priority
    df["priority_num"] = df["maintenance_priority"].map(PRIORITY_MAP).fillna(2)

    df = df.reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} maintenance records.")
    return df


def get_section_trains(trains_df, section_id):
    """Return only the trains that operate on the given section."""
    return trains_df[trains_df["section_id"] == section_id].copy()


if __name__ == "__main__":
    trains = load_trains()
    maint = load_maintenance()
    print(trains.head(3))
    print(maint.head(3))
