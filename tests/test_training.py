from math import isfinite

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from src.mp209.clean_data import clean_training_data, prepare_training_data
from src.mp209.train import build_model, chronological_folds


def test_cleaning_repairs_only_corrupted_values():
    raw = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-02-04", "2026-02-20", "2026-02-21", "2026-03-01"],
            "surge": [1.0, 9999.0, 2.0, 3.0, 2.0],
            "drivers_nearby": [4, 39, 39, 39, 20],
            "car_type": [" AUTO ", "aut.", "min.", "sed.", "Sedan"],
            "pickup_eta_min": [10.0, 11.5, 12.0, 9.0, 11.0],
        }
    )
    original = raw.copy()
    settings = {
        "sentinel_column": "surge",
        "sentinel_value": 9999,
        "stuck_sensor_column": "drivers_nearby",
        "stuck_start_date": "2026-02-04",
        "stuck_end_date": "2026-02-20",
    }
    prepared = prepare_training_data(raw, settings)
    assert pd.isna(prepared.loc[1, "surge"])
    assert prepared.loc[1:2, "drivers_nearby"].isna().all()

    cleaned = clean_training_data(prepared, settings)
    assert cleaned["surge"].tolist() == [1.0, 2.0, 2.0, 3.0, 2.0]
    assert cleaned["drivers_nearby"].tolist() == [4, 20, 20, 39, 20]
    assert cleaned["car_type"].tolist() == ["auto", "auto", "mini", "sedan", "sedan"]
    assert cleaned["pickup_eta_min"].equals(raw["pickup_eta_min"])
    assert_frame_equal(raw, original)


def test_chronological_folds_keep_booking_dates_together():
    dates = pd.Series(pd.date_range("2026-01-01", periods=6).repeat(2))
    folds = list(chronological_folds(dates, cv_splits=2))

    assert len(folds) == 2
    for train, validation in folds:
        assert dates.iloc[train].max() < dates.iloc[validation].min()
        assert len(train) % 2 == len(validation) % 2 == 0


def test_pipeline_predicts_with_missing_numeric_and_unseen_category():
    features = {
        "numerical": [
            "drivers_nearby",
            "distance_to_driver_km",
            "hour",
            "rain_mm",
            "surge",
            "road_closures",
        ],
        "categorical": ["car_type", "zone"],
    }
    bookings = pd.DataFrame(
        {
            "drivers_nearby": [8, 9, 10],
            "distance_to_driver_km": [1.0, 3.0, 5.0],
            "hour": [8, 8, 8],
            "rain_mm": [0.0, 0.0, 0.0],
            "surge": [1.0, 1.0, 1.0],
            "road_closures": [0, 0, 0],
            "car_type": ["mini", "auto", "sedan"],
            "zone": ["Airport", "Old City", "Tech Park"],
        }
    )
    pipeline = build_model(LinearRegression(), features)
    pipeline.fit(bookings, [5.0, 10.0, 15.0])
    new_booking = bookings.iloc[[0]].assign(surge=float("nan"), zone="Other")

    [prediction] = pipeline.predict(new_booking)
    assert isfinite(prediction)
