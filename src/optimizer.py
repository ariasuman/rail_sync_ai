"""
optimizer.py
------------
Scores every feasible maintenance window and returns the top 3.

Scoring formula (lower is better — like a cost function):
  score = w1 * norm(affected_trains)
        + w2 * norm(total_delay)
        + w3 * norm(predicted_traffic)
        - w4 * worker_available_bonus
        - w5 * machine_available_bonus
        - w6 * priority_bonus
        + w7 * deadline_urgency_penalty

The window with the LOWEST score is the best recommendation.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from conflict_detection import detect_conflicts, get_all_windows, minutes_to_hhmm
from traffic_prediction import predict_traffic

# Weights for each factor (must sum to a meaningful scale)
WEIGHTS = {
    "affected_trains": 0.30,
    "total_delay":     0.25,
    "traffic":         0.15,
    "worker":          0.10,   # bonus (subtracted)
    "machine":         0.10,   # bonus (subtracted)
    "priority":        0.05,   # bonus (subtracted)
    "deadline":        0.05,   # penalty (added)
}

PRIORITY_BONUS = {"Low": 0.0, "Medium": 0.25, "High": 0.75, "Critical": 1.0}


def _normalize(series):
    """Min-max normalize a pandas Series. Returns values in [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def compute_deadline_urgency(deadline_str, window_start_min):
    """
    Returns a penalty [0, 1] based on how close the deadline is.
    Closer deadline → higher urgency → higher penalty for NOT doing it now.
    We invert: if deadline is very close, windows far in the future get penalized.
    Here we simply return 0 (no penalty) since we're scheduling within today.
    For multi-day scheduling this would compare dates.
    """
    try:
        deadline = pd.to_datetime(deadline_str)
        days_left = (deadline - datetime.now()).days
        if days_left <= 0:
            return 1.0   # overdue — maximum urgency
        return max(0.0, 1.0 - days_left / 30.0)  # normalised over 30 days
    except Exception:
        return 0.0


def find_best_windows(
    trains_df,
    section_id,
    duration_hours,
    maintenance_priority,
    deadline,
    worker_available,
    machine_available,
    traffic_model=None,
    top_n=3,
    step_min=30,
):
    """
    Evaluate all candidate windows and return the top_n best ones.

    Parameters
    ----------
    trains_df           : cleaned trains DataFrame
    section_id          : railway section being maintained
    duration_hours      : how long the maintenance takes
    maintenance_priority: 'Low' / 'Medium' / 'High' / 'Critical'
    deadline            : deadline date string 'YYYY-MM-DD'
    worker_available    : bool
    machine_available   : bool
    traffic_model       : trained sklearn model (can be None)
    top_n               : number of top windows to return
    step_min            : granularity of window search in minutes

    Returns
    -------
    results_df : DataFrame with top_n windows sorted by score (ascending)
    """
    windows = get_all_windows(duration_hours, step_min)

    if not windows:
        return pd.DataFrame()

    rows = []
    for start_min, end_min in windows:
        affected, total_delay = detect_conflicts(
            trains_df, section_id, start_min, end_min
        )
        predicted_traffic = predict_traffic(traffic_model, start_min, end_min)

        rows.append({
            "start_min": start_min,
            "end_min": end_min,
            "start_time": minutes_to_hhmm(start_min),
            "end_time": minutes_to_hhmm(end_min),
            "affected_trains": len(affected),
            "total_delay_min": total_delay,
            "predicted_traffic": predicted_traffic,
            "worker_available": worker_available,
            "machine_available": machine_available,
            "maintenance_priority": maintenance_priority,
        })

    df = pd.DataFrame(rows)

    # Normalise cost components
    df["n_affected_norm"] = _normalize(df["affected_trains"])
    df["delay_norm"]      = _normalize(df["total_delay_min"])
    df["traffic_norm"]    = _normalize(df["predicted_traffic"])

    # Bonuses (reduce score when resources are available)
    worker_bonus  = 1.0 if worker_available  else 0.0
    machine_bonus = 1.0 if machine_available else 0.0
    p_bonus       = PRIORITY_BONUS.get(maintenance_priority, 0.25)

    # Deadline urgency (same for all windows in a single-day view)
    urgency = compute_deadline_urgency(deadline, 0)

    df["score"] = (
        WEIGHTS["affected_trains"] * df["n_affected_norm"]
        + WEIGHTS["total_delay"]   * df["delay_norm"]
        + WEIGHTS["traffic"]       * df["traffic_norm"]
        - WEIGHTS["worker"]        * worker_bonus
        - WEIGHTS["machine"]       * machine_bonus
        - WEIGHTS["priority"]      * p_bonus
        + WEIGHTS["deadline"]      * urgency
    )

    # Sort ascending (lowest score = best)
    df = df.sort_values("score").reset_index(drop=True)
    return df.head(top_n)


def explain_recommendation(row, trains_df, section_id):
    """
    Generate a human-readable explanation for why a window was chosen.
    """
    start = row["start_time"]
    end   = row["end_time"]
    n     = int(row["affected_trains"])
    delay = int(row["total_delay_min"])
    score = round(float(row["score"]), 4)
    worker  = "available" if row["worker_available"]  else "NOT available"
    machine = "available" if row["machine_available"] else "NOT available"
    priority = row["maintenance_priority"]

    reasons = []
    if n == 0:
        reasons.append("no trains are affected")
    elif n <= 2:
        reasons.append(f"only {n} train(s) are affected")
    else:
        reasons.append(f"{n} trains are affected (lowest feasible option)")

    if delay == 0:
        reasons.append("zero estimated delay")
    else:
        reasons.append(f"estimated delay is only {delay} minutes")

    reasons.append(f"maintenance worker is {worker}")
    reasons.append(f"machine is {machine}")
    reasons.append(f"maintenance priority is {priority}")

    reason_str = ", ".join(reasons)
    return (
        f"{start} to {end} is recommended because {reason_str}. "
        f"(Optimization score: {score})"
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_data import save_data
    from preprocess import load_trains
    from traffic_prediction import train_model

    save_data()
    trains = load_trains()
    model, mae, _ = train_model(trains, "DEL-AGR", 120)

    results = find_best_windows(
        trains_df=trains,
        section_id="DEL-AGR",
        duration_hours=2,
        maintenance_priority="High",
        deadline="2025-06-15",
        worker_available=True,
        machine_available=True,
        traffic_model=model,
        top_n=3,
    )
    print(results[["start_time", "end_time", "affected_trains",
                    "total_delay_min", "score"]])
    print()
    print(explain_recommendation(results.iloc[0], trains, "DEL-AGR"))
