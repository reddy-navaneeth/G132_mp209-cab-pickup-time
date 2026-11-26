import pandas as pd
import numpy as np
import yaml

with open("params.yaml", "r") as file:
    config = yaml.safe_load(file)

df = pd.read_csv(config["data"]["train_path"])

print("Original shape:", df.shape)

# Handle sentinel values
# The following operation converts invalid surge values into missing values:

df["surge"] = df["surge"].replace(9999, np.nan)
surge_median = df["surge"].median()
print("Surge median:", surge_median)
df["surge"] = df["surge"].fillna(surge_median)


# Handle the stuck sensor
df["date"] = pd.to_datetime(df["date"])

print(
    df.loc[
        df["date"].between("2026-02-01", "2026-02-23")
    ].groupby("date")["drivers_nearby"].agg(
        ["min", "max", "count"]
    )
)


# ategorical cleaning

df["car_type"] = df["car_type"].str.strip().str.lower()


# Create a mask for the affected dates

stuck_mask = df["date"].between(
    "2026-02-04", "2026-02-20"
)

print("Stuck sensor affected rows:", stuck_mask.sum())

# Create a mask for the affected dates
drivers_median = df.loc[
    ~stuck_mask, "drivers_nearby"
].median()

print("Drivers median:", drivers_median)

# Replace the corrupted readings

df.loc[stuck_mask, "drivers_nearby"] = int(
    round(drivers_median)
)




# 
# Fix spelling variants
df["car_type"] = df["car_type"].str.strip().str.lower()
car_mapping = {
    "aut.": "auto",
    "min.": "mini",
    "sed.": "sedan"
}

df["car_type"] = df["car_type"].replace(car_mapping)

print("\nCleaned car categories:")
print(df["car_type"].value_counts())



# Validate the cleaned dataset

print("\n--- CLEANED DATA VALIDATION ---")

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nRemaining sentinel values:")
print((df["surge"] == 9999).sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique car categories:")
print(df["car_type"].unique())

print("\nCleaned statistical summary:")
print(df.describe().to_string())




# save the cleaned dataset 
df.to_csv(
    "data/processed/cleaned_train.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")