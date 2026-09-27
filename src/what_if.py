"""
what_if.py
----------
Two functions:
  1. simulate_window()   — user proposes a custom window; system evaluates it.
  2. replan()            — accepts updated train schedules and re-runs optimization.
"""

import pandas as pd
from conflict_detection import detect_conflicts, minutes_to_hhmm
from optimizer import find_best_windows, explain_recommendation
from traffic_prediction import predict_traffic


def simulate_window(
    trains_df,
    section_id,
    start_hhmm,
    end_hhmm,
    maintenance_priority,
    deadline,
    worker_available,
    machine_available,
    traffic_model=None,
):
    """
    Evaluate a user-proposed maintenance window.

    Parameters
    ----------
    start_hhmm / end_hhmm : strings like '02:00', '04:00'

    Returns
    -------
    dict with keys: start_time, end_time, affected_trains, total_delay_min,
                    predicted_traffic, score, feasible, explanation
    """
    # Parse times
    try:
        sh, sm = map(int, start_hhmm.split(":"))
        eh, em = map(int, end_hhmm.split(":"))
        start_min = sh * 60 + sm
        end_min   = eh * 60 + em
        if end_min <= start_min:
            end_min += 1440   # overnight window
    except ValueError:
        return {"error": "Invalid time format. Use HH:MM."}

    affected, total_delay = detect_conflicts(
        trains_df, section_id, start_min, end_min
    )
    predicted_traffic = predict_traffic(traffic_model, start_min, end_min)

    # Feasibility: worker AND machine must be available
    feasible = worker_available and machine_available

    # Simple score (same formula as optimizer, single window)
    from optimizer import WEIGHTS, PRIORITY_BONUS, compute_deadline_urgency
    urgency = compute_deadline_urgency(deadline, start_min)
    p_bonus = PRIORITY_BONUS.get(maintenance_priority, 0.25)

    # Without normalisation (single window), use raw values scaled to [0,1] heuristically
    n_norm = min(len(affected) / 10.0, 1.0)
    d_norm = min(total_delay / 300.0, 1.0)
    t_norm = min(predicted_traffic / 10.0, 1.0)

    score = (
        WEIGHTS["affected_trains"] * n_norm
        + WEIGHTS["total_delay"]   * d_norm
        + WEIGHTS["traffic"]       * t_norm
        - WEIGHTS["worker"]        * (1.0 if worker_available  else 0.0)
        - WEIGHTS["machine"]       * (1.0 if machine_available else 0.0)
        - WEIGHTS["priority"]      * p_bonus
        + WEIGHTS["deadline"]      * urgency
    )

    worker_str  = "available" if worker_available  else "NOT available"
    machine_str = "available" if machine_available else "NOT available"
    n = len(affected)
    reasons = []
    if n == 0:
        reasons.append("no trains are affected")
    else:
        reasons.append(f"{n} train(s) are affected with {total_delay} min estimated delay")
    reasons.append(f"worker is {worker_str}, machine is {machine_str}")
    if not feasible:
        reasons.append("⚠️ window is NOT feasible due to resource unavailability")

    explanation = (
        f"{start_hhmm}–{end_hhmm}: {', '.join(reasons)}. "
        f"Score: {round(score, 4)}."
    )

    return {
        "start_time": start_hhmm,
        "end_time": end_hhmm,
        "affected_trains": n,
        "affected_train_list": affected[
            ["train_id", "train_name", "train_priority", "estimated_delay_min"]
        ].to_dict("records") if n > 0 else [],
        "total_delay_min": total_delay,
        "predicted_traffic": round(predicted_traffic, 2),
        "score": round(score, 4),
        "feasible": feasible,
        "explanation": explanation,
    }


def replan(
    updated_trains_df,
    section_id,
    duration_hours,
    maintenance_priority,
    deadline,
    worker_available,
    machine_available,
    traffic_model=None,
    top_n=3,
):
    """
    Re-run the full optimization with an updated train schedule.
    Useful when new trains are added or schedules change.

    Returns top_n best windows as a DataFrame.
    """
    print(f"[RE-PLAN] Re-planning for section {section_id} with {len(updated_trains_df)} trains...")
    results = find_best_windows(
        trains_df=updated_trains_df,
        section_id=section_id,
        duration_hours=duration_hours,
        maintenance_priority=maintenance_priority,
        deadline=deadline,
        worker_available=worker_available,
        machine_available=machine_available,
        traffic_model=traffic_model,
        top_n=top_n,
    )
    return results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_data import save_data
    from preprocess import load_trains
    from traffic_prediction import train_model

    save_data()
    trains = load_trains()
    model, _, _ = train_model(trains, "DEL-AGR", 120)

    # What-if: test 02:00–04:00
    result = simulate_window(
        trains, "DEL-AGR", "02:00", "04:00",
        "High", "2025-06-15", True, True, model
    )
    print(result)

    # Re-plan with a modified schedule (add a new train)
    import pandas as pd
    new_train = pd.DataFrame([{
        "train_id": "T9999", "train_name": "Test Express",
        "section_id": "DEL-AGR", "arrival_time": "02:30",
        "departure_time": "03:30", "train_priority": "Critical",
        "day_of_week": 1, "is_weekend": 0,
        "arr_min": 150, "dep_min": 210, "priority_num": 4,
    }])
    updated = pd.concat([trains, new_train], ignore_index=True)
    top = replan(updated, "DEL-AGR", 2, "High", "2025-06-15", True, True, model)
    print(top[["start_time", "end_time", "affected_trains", "score"]])
