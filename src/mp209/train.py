import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .clean_data import clean_training_data, prepare_training_data


def build_model(estimator, features):
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), features["numerical"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                features["categorical"],
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", estimator)])


def chronological_folds(dates, cv_splits):
    """Use expanding windows and keep every row from a date in the same split."""
    unique_dates = pd.Index(dates.unique()).sort_values()
    for train_indices, validation_indices in TimeSeriesSplit(cv_splits).split(unique_dates):
        yield (
            np.flatnonzero(dates.isin(unique_dates[train_indices])),
            np.flatnonzero(dates.isin(unique_dates[validation_indices])),
        )


def train_v1(config, data_sha256):
    data = config["data"]
    features = config["features"]
    feature_names = features["numerical"] + features["categorical"]
    target = data["target"]
    settings = config["training"]
    seed = settings["random_state"]

    raw = pd.read_csv(data["train_path"]).sort_values(["date", "record_id"])
    prepared = prepare_training_data(raw, config["cleaning"])
    split_index = int(len(prepared) * (1 - settings["validation_fraction"]))
    validation_start = prepared["date"].iloc[split_index]
    development = prepared.loc[prepared["date"] < validation_start]
    validation = prepared.loc[prepared["date"] >= validation_start]
    development_x, development_y = development[feature_names], development[target]
    validation_x, validation_y = validation[feature_names], validation[target]
    folds = list(chronological_folds(development["date"], settings["cv_splits"]))

    candidates = {
        "median-baseline": DummyRegressor(strategy="median"),
        "linear-regression": LinearRegression(),
        "decision-tree": DecisionTreeRegressor(
            max_depth=settings["tree_max_depth"],
            random_state=seed,
        ),
        "random-forest": RandomForestRegressor(
            n_estimators=settings["forest_n_estimators"],
            max_depth=settings["forest_max_depth"],
            random_state=seed,
        ),
    }
    baseline = candidates["median-baseline"]
    baseline_mae = mean_absolute_error(
        validation_y,
        baseline.fit(development_x, development_y).predict(validation_x),
    )

    tracking = config["tracking"]
    mlflow.set_experiment(tracking["experiment_name"])
    results = []
    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "data_sha256": data_sha256,
                    "model_type": type(estimator).__name__,
                    "cv_splits": settings["cv_splits"],
                    "development_rows": len(development),
                    "validation_rows": len(validation),
                    "validation_start": validation_start.date().isoformat(),
                    **estimator.get_params(),
                }
            )
            pipeline = build_model(estimator, features)
            scores = cross_validate(
                pipeline,
                development_x,
                development_y,
                cv=folds,
                scoring="neg_mean_absolute_error",
                return_train_score=True,
                error_score="raise",
            )
            cv_mae = float(-scores["test_score"].mean())
            mlflow.log_metrics(
                {
                    "mae_cv": cv_mae,
                    "mae_cv_std": float(scores["test_score"].std()),
                    "mae_cv_train": float(-scores["train_score"].mean()),
                }
            )
            results.append((cv_mae, name, run.info.run_id))
            print(f"{name}: {cv_mae:.3f} min CV MAE")

    _, winner, run_id = min(results)
    pipeline = build_model(candidates[winner], features)
    pipeline.fit(development_x, development_y)
    validation_mae = mean_absolute_error(validation_y, pipeline.predict(validation_x))
    cleaned = clean_training_data(prepared, config["cleaning"])
    pipeline.fit(cleaned[feature_names], cleaned[target])

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {"mae_validation": validation_mae, "mae_baseline_validation": baseline_mae}
        )
        mlflow.log_param("final_training_rows", len(cleaned))
        logged_model = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=cleaned[feature_names].head(3),
            serialization_format="cloudpickle",
        )
        version = mlflow.register_model(logged_model.model_uri, tracking["registered_model_name"])

    print(
        f"{winner} won. Validation MAE {validation_mae:.3f} min "
        f"(median baseline {baseline_mae:.3f} min)."
    )
    return version, pipeline


def main():
    config = yaml.safe_load(Path("params.yaml").read_text())
    tracking = config["tracking"]
    model_name = tracking["registered_model_name"]
    model_path = Path(tracking["model_path"])
    findings_path = Path(tracking["findings_path"])
    findings = yaml.safe_load(findings_path.read_text())
    data_sha256 = hashlib.sha256(Path(config["data"]["train_path"]).read_bytes()).hexdigest()

    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")

    if versions:
        if len(versions) != 1 or int(versions[0].version) != 1:
            raise SystemExit("The registry already contains a later model version.")
        version = versions[0]
        if client.get_run(version.run_id).data.params["data_sha256"] != data_sha256:
            raise SystemExit("The registered v1 was trained on a different source file.")
        pipeline = None
        print(f"Reusing {model_name} version 1")
    else:
        version, pipeline = train_v1(config, data_sha256)

    if pipeline is not None or not model_path.exists():
        if pipeline is None:
            pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/{version.version}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_tmp = model_path.with_suffix(".joblib.tmp")
        joblib.dump(pipeline, model_tmp)
        model_tmp.replace(model_path)

    client.set_registered_model_alias(model_name, "champion", version.version)
    findings["tracking"] = {
        "experiment_name": tracking["experiment_name"],
        "best_run_id": version.run_id,
        "registered_model_name": model_name,
        "champion_version": int(version.version),
    }
    findings_tmp = findings_path.with_suffix(".yaml.tmp")
    findings_tmp.write_text(yaml.safe_dump(findings, sort_keys=False))
    findings_tmp.replace(findings_path)
    print(f"{model_name}@champion points to version {version.version}, run {version.run_id}")


if __name__ == "__main__":
    main()
