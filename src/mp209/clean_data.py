from pathlib import Path

import pandas as pd
import yaml


def prepare_training_data(raw, settings):
    """Identify corrupt readings and normalize car types without fitting statistics."""
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])

    sentinel = settings["sentinel_column"]
    data[sentinel] = data[sentinel].mask(data[sentinel].eq(settings["sentinel_value"]))

    stuck = settings["stuck_sensor_column"]
    affected = data["date"].between(
        settings["stuck_start_date"],
        settings["stuck_end_date"],
    )
    data[stuck] = data[stuck].mask(affected)

    data["car_type"] = (
        data["car_type"]
        .str.strip()
        .str.lower()
        .replace({"aut.": "auto", "min.": "mini", "sed.": "sedan"})
    )
    return data


def clean_training_data(prepared, settings):
    """Fill corrupt readings using medians from the full training dataset."""
    data = prepared.copy()
    sentinel = settings["sentinel_column"]
    stuck = settings["stuck_sensor_column"]
    data[sentinel] = data[sentinel].fillna(data[sentinel].median())
    data[stuck] = data[stuck].fillna(round(data[stuck].median())).astype("int64")
    return data


def main():
    config = yaml.safe_load(Path("params.yaml").read_text())
    raw = pd.read_csv(config["data"]["train_path"])
    prepared = prepare_training_data(raw, config["cleaning"])
    cleaned = clean_training_data(prepared, config["cleaning"])
    output_path = Path(config["data"]["cleaned_train_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Wrote {len(cleaned)} cleaned rows to {output_path}")


if __name__ == "__main__":
    main()
