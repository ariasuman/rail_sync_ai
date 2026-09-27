"""
traffic_prediction.py
---------------------
Trains a Random Forest model to predict how many trains are expected
in a given time window on a given section.

Features used:
  - window_start_min : start of window in minutes since midnight
  - window_end_min   : end of window in minutes since midnight
  - hour_of_day      : starting hour (0-23)
  - is_night         : 1 if window starts between 22:00 and 05:00
  - duration_min     : length of the window in minutes

Target:
  - train_count : number of trains whose schedule overlaps this window
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def build_training_data(trains_df, section_id, window_size_min=120, step_min=30):
    """
    Slide a window across 24 hours and count how many trains overlap each window.
    Returns a DataFrame of features + target for model training.
    """
    section_trains = trains_df[trains_df["section_id"] == section_id]

    rows = []
    # Slide window from 00:00 to 23:30
    for start in range(0, 1440, step_min):
        end = start + window_size_min
        # Count trains whose [arr_min, dep_min] overlaps [start, end]
        overlap = section_trains[
            (section_trains["arr_min"] < end) &
            (section_trains["dep_min"] > start)
        ]
        count = len(overlap)
        hour = start // 60
        rows.append({
            "window_start_min": start,
            "window_end_min": min(end, 1440),
            "hour_of_day": hour,
            "is_night": int(hour >= 22 or hour <= 5),
            "duration_min": window_size_min,
            "train_count": count,
        })

    return pd.DataFrame(rows)


def train_model(trains_df, section_id, window_size_min=120):
    """
    Train a Random Forest on sliding-window traffic data for a section.
    Returns the trained model and MAE on test set.
    """
    df = build_training_data(trains_df, section_id, window_size_min)

    if len(df) < 10:
        # Not enough data — return a dummy model
        return None, None, df

    features = ["window_start_min", "window_end_min", "hour_of_day",
                "is_night", "duration_min"]
    X = df[features]
    y = df["train_count"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae, df


def predict_traffic(model, start_min, end_min):
    """
    Predict expected train count for a given window using the trained model.
    Falls back to 0 if model is None.
    """
    if model is None:
        return 0.0
    hour = start_min // 60
    X = pd.DataFrame([{
        "window_start_min": start_min,
        "window_end_min": end_min,
        "hour_of_day": hour,
        "is_night": int(hour >= 22 or hour <= 5),
        "duration_min": end_min - start_min,
    }])
    return float(model.predict(X)[0])


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_data import save_data
    from preprocess import load_trains

    save_data()
    trains = load_trains()
    model, mae, df = train_model(trains, "DEL-AGR", window_size_min=120)
    print(f"MAE: {mae:.2f}")
    print(df.head())
