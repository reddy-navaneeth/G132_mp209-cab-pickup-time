# MP209 – Cab Pickup Time Prediction (Kochi)

**Group ID:** G132  
**Problem ID:** MP209

## Team Members

| Name | Enrollment Number |
|---|---|
| Vanshika Ahlawat | S24CSEU1205 |
| Daksh | S24CSEU1177 |
| Tanush Sharma | S24CSEU1169 |
| Reddy Narayana Navaneeth | S24CSEU1183 |

## Train and verify v1

Use Python 3.12 and run these commands from the repository root. Restore the
original `train.csv` and `later.csv` from the assignment package into
`data/raw/`. The v1 trainer reads only `train.csv`.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.mp209.clean_data
python -m src.mp209.train
pytest
```

Training compares a median baseline, linear regression, a shallow decision
tree, and a random forest using five expanding-window folds. The latest
roughly 20% of the January–June records provide a separate validation score.
The winning pipeline is then fitted to all cleaned January–June rows.

MLflow records the training CSV hash and MAE for every run in `mlflow.db`.
The winner is registered as `mp209-pickup-eta@champion` (version 1), exported
to `models/model.joblib`, and recorded in `reports/findings.yaml`. Running the
trainer again with the same data and registry reuses version 1.

Run the MLflow UI from the repository root.

```sh
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
