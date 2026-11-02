import sys
import argparse
import pandas as pd
import numpy as np

# ----------------------------
# HELPERS: GEOGRAPHY & NORMALIZATION
# ----------------------------

def normalize_dataset_type(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if s == "ignition":
        return "ignition"
    if s in {"nonignition", "noignition", "nonfire"}:
        return "nonignition"
    return None

def in_alberta(lat, lon):
    return (49.0 <= lat <= 60.0 and -120.0 <= lon <= -110.0)

def in_bc(lat, lon):
    """BC: 49-60 N, 139.5-115 W (excludes Alberta territory)"""
    return (49.0 <= lat <= 60.0 and -139.5 <= lon < -115.0)

def assign_province(lat, lon):
    if pd.isna(lat) or pd.isna(lon):
        return None
    if in_alberta(lat, lon):
        return "Alberta"
    elif in_bc(lat, lon):
        return "British Columbia"
    else:
        return None

# ----------------------------
# MAIN DIAGNOSTIC FUNCTION
# ----------------------------

def run_diagnostics(csv_path):
    print(f"\n{'='*20} ANALYZING: {csv_path} {'='*20}")

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 1. Dataset Dimensions
    print(f"\n[1] DATASET DIMENSIONS")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")

    # 2. Class Balance & Corrupted Label Check
    print(f"\n[2] CLASS DISTRIBUTION (label_ignition)")
    label_counts = df['label_ignition'].value_counts(dropna=False)
    print(label_counts.to_string())

    ign_count = label_counts.get(1.0, 0)
    non_ign_count = label_counts.get(0.0, 0)

    if ign_count > 0:
        ratio = non_ign_count / ign_count
        print(f"  - Calculated Ratio: 1:{ratio:.2f}")

    # Any label other than 0 or 1 is corrupted
    corrupted = label_counts[~label_counts.index.isin([0.0, 1.0])].index.tolist()
    if corrupted:
        print(f"  !! WARNING: Corrupted labels found: {corrupted} !!")

    # 3. Geographic & Dataset Type Audit
    print(f"\n[3] PROVINCIAL & DATASET TYPE BREAKDOWN")

    # Create temp columns for analysis without modifying original data for training
    temp_df = df.copy()

    # Handle label mapping if dataset_type is missing but label_ignition exists
    if "dataset_type" not in temp_df.columns and "label_ignition" in temp_df.columns:
        temp_df["dataset_type_norm"] = temp_df["label_ignition"].map({1: "ignition", 0: "nonignition"})
    elif "dataset_type" in temp_df.columns:
        temp_df["dataset_type_norm"] = temp_df["dataset_type"].apply(normalize_dataset_type)

    if "strike_lat" in temp_df.columns and "strike_lon" in temp_df.columns:
        temp_df["province"] = temp_df.apply(
            lambda row: assign_province(row["strike_lat"], row["strike_lon"]), axis=1
        )

        for prov in ["Alberta", "British Columbia"]:
            for d_type in ["ignition", "nonignition"]:
                count = len(temp_df[(temp_df["province"] == prov) & (temp_df["dataset_type_norm"] == d_type)])
                print(f"  - {d_type.capitalize()} in {prov}: {count:,}")
    else:
        print("  - Skipping Provincial check: 'strike_lat' or 'strike_lon' missing.")

    # 4. Critical Feature Health
    print(f"\n[4] CRITICAL FEATURE HEALTH")
    features_to_check = [
        "volumetric_soil_water_layer_1_lag_0",
        "moisture_drying_rate",
        "thermal_stress",
        "hot_dry_index"
    ]
    for feat in features_to_check:
        if feat in df.columns:
            missing = df[feat].isna().sum()
            print(f"  - {feat}: {missing:,} NaNs ({(missing/len(df))*100:.2f}%)")
        else:
            print(f"  - !! {feat} IS MISSING !!")

    # 5. Schema Preview
    print(f"\n[5] COLUMN LIST PREVIEW")
    print(list(df.columns))

    print(f"\n{'='*20} DIAGNOSTIC COMPLETE {'='*20}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full diagnostic suite for Wildfire Ignition Datasets")
    parser.add_argument("csv_path", help="Path to the preprocessed CSV file")
    args = parser.parse_args()

    run_diagnostics(args.csv_path)
