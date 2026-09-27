"""
generate_data.py
----------------
Generates synthetic railway train schedules and maintenance records.
Saves them as CSV files in the data/ folder.
Run this once before using the rest of the system.
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────
SECTIONS = ["DEL-AGR", "MUM-PUN", "CHN-BLR", "HYD-SEC", "KOL-PAT",
            "DEL-JAI", "LKO-CNB", "BPL-NGP", "AMD-BRC", "MAS-CBE"]

TRAIN_NAMES = [
    "Rajdhani Express", "Shatabdi Express", "Duronto Express",
    "Garib Rath", "Jan Shatabdi", "Humsafar Express",
    "Tejas Express", "Vande Bharat", "Intercity Express",
    "Mail Express", "Passenger Train", "Goods Train",
    "Double Decker", "Sampark Kranti", "Superfast Express"
]

ASSET_TYPES = ["Track", "Signal", "Bridge", "Overhead Wire", "Switch/Points",
               "Level Crossing", "Platform", "Traction Substation"]

MAINTENANCE_TYPES = ["Routine Inspection", "Track Tamping", "Rail Replacement",
                     "Signal Calibration", "Bridge Inspection", "OHE Maintenance",
                     "Points Lubrication", "Ballast Cleaning"]

PRIORITIES = ["Low", "Medium", "High", "Critical"]


def random_time(hour_start=0, hour_end=23):
    """Return a random HH:MM time string within a given hour range."""
    h = random.randint(hour_start, hour_end)
    m = random.choice([0, 15, 30, 45])
    return f"{h:02d}:{m:02d}"


def generate_trains(n=120):
    """
    Generate n synthetic train schedule records.
    Each record represents one train passing through one section on one day.
    """
    records = []
    base_date = datetime(2025, 6, 1)

    for i in range(n):
        section = random.choice(SECTIONS)
        arr_hour = random.randint(0, 22)
        arr_min = random.choice([0, 15, 30, 45])
        arrival = datetime(2025, 6, 1, arr_hour, arr_min)

        # Travel time between 30 min and 3 hours
        travel_minutes = random.choice([30, 45, 60, 90, 120, 150, 180])
        departure = arrival + timedelta(minutes=travel_minutes)

        # Wrap departure to next day if needed
        dep_str = departure.strftime("%H:%M")
        arr_str = arrival.strftime("%H:%M")

        priority = random.choices(
            PRIORITIES, weights=[20, 40, 30, 10]
        )[0]

        records.append({
            "train_id": f"T{1000 + i}",
            "train_name": random.choice(TRAIN_NAMES),
            "section_id": section,
            "arrival_time": arr_str,
            "departure_time": dep_str,
            "train_priority": priority,
            "day_of_week": random.randint(0, 6),   # 0=Mon, 6=Sun
            "is_weekend": int(random.randint(0, 6) >= 5),
        })

    return pd.DataFrame(records)


def generate_maintenance(n=30):
    """
    Generate n synthetic maintenance task records.
    Each record is one maintenance job that needs to be scheduled.
    """
    records = []
    base_date = datetime(2025, 6, 1)

    for i in range(n):
        section = random.choice(SECTIONS)
        asset_type = random.choice(ASSET_TYPES)
        maint_type = random.choice(MAINTENANCE_TYPES)

        # Asset condition: 1 (critical) to 5 (good)
        condition = random.randint(1, 5)

        # Worse condition → higher maintenance priority
        if condition <= 2:
            m_priority = random.choice(["High", "Critical"])
        elif condition == 3:
            m_priority = random.choice(["Medium", "High"])
        else:
            m_priority = random.choice(["Low", "Medium"])

        # Deadline: 3 to 30 days from base date
        deadline = base_date + timedelta(days=random.randint(3, 30))

        records.append({
            "asset_id": f"A{200 + i}",
            "asset_type": asset_type,
            "section_id": section,
            "asset_condition": condition,
            "maintenance_type": maint_type,
            "maintenance_duration_hours": random.choice([1, 2, 3, 4, 6, 8]),
            "maintenance_priority": m_priority,
            "maintenance_deadline": deadline.strftime("%Y-%m-%d"),
            "worker_available": random.choice([True, False, True, True]),  # 75% available
            "machine_available": random.choice([True, False, True, True]),
        })

    return pd.DataFrame(records)


def save_data():
    """Generate and save both CSVs to the data/ folder."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    trains_df = generate_trains(120)
    maint_df = generate_maintenance(30)

    trains_path = os.path.join(out_dir, "trains.csv")
    maint_path = os.path.join(out_dir, "maintenance.csv")

    trains_df.to_csv(trains_path, index=False)
    maint_df.to_csv(maint_path, index=False)

    print(f"[OK] Saved {len(trains_df)} train records -> {trains_path}")
    print(f"[OK] Saved {len(maint_df)} maintenance records -> {maint_path}")
    return trains_df, maint_df


if __name__ == "__main__":
    save_data()
