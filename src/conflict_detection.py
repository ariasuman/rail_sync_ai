"""
conflict_detection.py
---------------------
For a given maintenance window (start_min, end_min) on a section,
finds all trains that overlap and estimates the total delay caused.

Delay logic:
  - Critical train delayed: 60 min penalty
  - High priority train:    30 min penalty
  - Medium priority train:  15 min penalty
  - Low priority train:      5 min penalty
"""

import pandas as pd

DELAY_PENALTY = {"Critical": 60, "High": 30, "Medium": 15, "Low": 5}


def detect_conflicts(trains_df, section_id, start_min, end_min):
    """
    Find trains on `section_id` whose schedule overlaps [start_min, end_min].

    Returns:
        affected_trains (DataFrame) : rows of conflicting trains
        total_delay_min (int)       : estimated total delay in minutes
    """
    section_trains = trains_df[trains_df["section_id"] == section_id].copy()

    # Overlap condition: train arrives before window ends AND departs after window starts
    affected = section_trains[
        (section_trains["arr_min"] < end_min) &
        (section_trains["dep_min"] > start_min)
    ].copy()

    # Estimate delay per train based on priority
    affected["estimated_delay_min"] = affected["train_priority"].map(
        DELAY_PENALTY
    ).fillna(15)

    total_delay = int(affected["estimated_delay_min"].sum())
    return affected, total_delay


def get_all_windows(duration_hours, step_min=30):
    """
    Generate all candidate maintenance windows across 24 hours.
    Returns list of (start_min, end_min) tuples.
    """
    duration_min = duration_hours * 60
    windows = []
    for start in range(0, 1440 - duration_min + 1, step_min):
        windows.append((start, start + duration_min))
    return windows


def minutes_to_hhmm(minutes):
    """Convert integer minutes-since-midnight to 'HH:MM' string."""
    minutes = int(minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_data import save_data
    from preprocess import load_trains

    save_data()
    trains = load_trains()
    affected, delay = detect_conflicts(trains, "DEL-AGR", 120, 240)
    print(f"Affected trains: {len(affected)}, Total delay: {delay} min")
    print(affected[["train_id", "train_name", "train_priority", "estimated_delay_min"]])
