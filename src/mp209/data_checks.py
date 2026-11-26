import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\n--- DATASET SHAPES ---")
print("Train:", train.shape)
print("Later:", later.shape)
print("Holdout:", holdout.shape)

print("\n--- FIRST FIVE ROWS ---")
print(train.head().to_string())

print("\n--- COLUMN NAMES ---")
print(train.columns.tolist())

print("\n--- DATA TYPES ---")
print(train.dtypes)

print("\n--- MISSING VALUES ---")
print(train.isnull().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(train.describe().to_string())

print("\n--- DUPLICATE ROWS ---")
print(train.duplicated().sum())



print("\n--- CHECK 1: DATA LEAKAGE ---")

extra_columns = set(train.columns) - set(holdout.columns)

print("Columns absent from holdout:", extra_columns)

leaking_columns = extra_columns - {"pickup_eta_min"}

print("Potential leaking columns:", leaking_columns)





print("\n--- CHECK 2: UNIT MIX-UP ---")

numeric_features = [
    "drivers_nearby",
    "distance_to_driver_km",
    "hour",
    "rain_mm",
    "surge",
    "road_closures"
]

for column in numeric_features:
    print(f"\nFeature: {column}")
    print(
        train.groupby("zone")[column]
        .agg(["min", "mean", "median", "max"])
    )





    print("\n--- CHECK 3: SENTINEL VALUES ---")

sentinels = [-1, -999, 999, 9999]

for column in numeric_features:
    for value in sentinels:
        count = (train[column] == value).sum()

        if count > 0:
            print("Column:", column)
            print("Sentinel value:", value)
            print("Count:", count)







print("\n--- CHECK 4: DUPLICATE ROWS ---")

print("Duplicate count:", train.duplicated().sum())            




print("\n--- CHECK 5: STUCK SENSOR ---")

for column in numeric_features:
    daily = train.groupby("date")[column].agg(
        ["min", "max", "nunique", "count"]
    )

    constant_days = daily[
        (daily["nunique"] == 1) &
        (daily["count"] > 1)
    ]

    print(f"\nFeature: {column}")
    print("Constant days:")
    print(constant_days.to_string())















    print("\n--- CHECK 6: TARGET TYPOS ---")

target = "pickup_eta_min"

print("Largest 20 target values:")
print(train[target].nlargest(20).to_string())

print("\nTraining target statistics:")
print(train[target].describe())

print("\nLater target statistics:")
print(later[target].describe())

sorted_target = train[target].sort_values()

print("\nLargest gaps between target values:")
print(sorted_target.diff().nlargest(10))





print("\n--- CHECK 7: SPELLING VARIANTS ---")

categorical_features = ["car_type", "zone"]

for column in categorical_features:
    print(f"\nFeature: {column}")

    print("Training categories:")
    print(train[column].value_counts())

    print("Exact training category names:")
    print([repr(v) for v in train[column].unique()])

    print("Later categories:")
    print(later[column].value_counts())

    print("Unique training levels:")
    print(train[column].nunique())

    print("Unique later levels:")
    print(later[column].nunique())