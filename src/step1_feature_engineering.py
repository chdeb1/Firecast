import os
import argparse
import pandas as pd
import numpy as np

from common import (
    RAW_FILES_ALL, RAW_FILES_14_TO_18, TRAIN_AB_CSV, TRAIN_BC_CSV,
    PROCESSED_DIR, OSM_CACHE_DIR, OSMProvinceAssigner,
)

# ----------------------------
# CONFIG
# ----------------------------
INPUT_SETS = {
    "14_to_18": RAW_FILES_14_TO_18,
    "all": RAW_FILES_ALL,
}

OUTPUT_TRAIN_AB = str(TRAIN_AB_CSV)
OUTPUT_TRAIN_BC = str(TRAIN_BC_CSV)

# Fraction of non-ignition rows kept
TRAIN_DOWNSAMPLE_RATIO = 1.0

CHUNK_SIZE = 500_000
RANDOM_SEED = 42

# ----------------------------
# HELPERS
# ----------------------------

def ensure_strike_fields(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    rename_map = {
        "Latitude": "strike_lat", "Longitude": "strike_lon",
        "Peak Signal (kiloAmps)": "peak_signal_kA", "Cloud or Ground": "cloud_or_ground"
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "strike_datetime" in df.columns:
        df["strike_datetime"] = pd.to_datetime(df["strike_datetime"], errors='coerce')

    return df.dropna(subset=["strike_datetime"])

def assign_provinces_to_chunk(df, assigner):
    """Assign provinces to a chunk of data using OSM boundaries"""
    if "strike_lat" not in df.columns or "strike_lon" not in df.columns:
        print("  WARNING: Missing coordinate columns, cannot assign provinces")
        return None, None

    df["province"] = assigner.assign(df["strike_lat"], df["strike_lon"])

    # Split by province
    chunk_ab = df[df["province"] == "Alberta"].drop(columns=["province"])
    chunk_bc = df[df["province"] == "British Columbia"].drop(columns=["province"])

    return chunk_ab, chunk_bc

def engineer_chunk_minimal(df):
    # Moisture/Thermal Interactions
    if "volumetric_soil_water_layer_1_lag_0" in df.columns and "volumetric_soil_water_layer_1_lag_24" in df.columns:
        df["moisture_drying_rate"] = df["volumetric_soil_water_layer_1_lag_24"] - df["volumetric_soil_water_layer_1_lag_0"]
    if "skin_temperature_lag_0" in df.columns and "2m_temperature_lag_0" in df.columns:
        df["thermal_stress"] = df["skin_temperature_lag_0"] - df["2m_temperature_lag_0"]
    if "skin_temperature_lag_0" in df.columns and "soil_temperature_level_4_lag_0" in df.columns:
        df["thermal_gradient_proxy"] = df["skin_temperature_lag_0"] - df["soil_temperature_level_4_lag_0"]

    if "skin_temperature_lag_0" in df.columns and "volumetric_soil_water_layer_1_lag_0" in df.columns:
        df["hot_dry_index"] = df["skin_temperature_lag_0"] / (df["volumetric_soil_water_layer_1_lag_0"] + 0.01)

    if "10m_u_component_of_wind_lag_0" in df.columns and "10m_v_component_of_wind_lag_0" in df.columns:
        wind_speed = np.sqrt(df["10m_u_component_of_wind_lag_0"]**2 + df["10m_v_component_of_wind_lag_0"]**2)
        df["evaporative_wind_stress"] = df["skin_temperature_lag_0"] * wind_speed
    return df

def apply_final_drops(df):
    """ONLY drops specific lags to clean feature set, preserves everything else."""
    lags_to_drop = [col for col in df.columns if any(s in col for s in ["_lag_1", "_lag_2", "_lag_24"])]
    lags_to_drop = [c for c in lags_to_drop if c != "label_ignition"]
    return df.drop(columns=lags_to_drop, errors='ignore')

# ----------------------------
# MAIN PROCESSING
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature engineering + AB/BC split")
    parser.add_argument("--inputs", choices=sorted(INPUT_SETS), default="14_to_18",
                        help="Which raw input files to process (default: 14_to_18)")
    parser.add_argument("--nonignition-frac", type=float, default=TRAIN_DOWNSAMPLE_RATIO,
                        help="Fraction of non-ignition rows to keep (default: 1.0 = all)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Only read the first N rows of each input file")
    args = parser.parse_args()

    input_files = [str(p) for p in INPUT_SETS[args.inputs]]
    downsample_ratio = args.nonignition_frac
    print(f"Inputs: {args.inputs} | non-ignition fraction kept: {downsample_ratio} | nrows: {args.nrows}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Clear output files if they exist
    for f in [OUTPUT_TRAIN_AB, OUTPUT_TRAIN_BC]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleared {f}")

    # Initialize OSM province assigner
    print("\n" + "="*70)
    print("INITIALIZING OPENSTREETMAP PROVINCE ASSIGNER")
    print("="*70)
    assigner = OSMProvinceAssigner(OSM_CACHE_DIR)

    # We determine col_order dynamically based on the first chunk processed
    col_order = None
    ab_file_exists = False
    bc_file_exists = False

    # Track stats for both regions
    total_input_rows = 0
    total_ab_rows = 0
    total_bc_rows = 0
    total_other_rows = 0

    print("\n" + "="*70)
    print("PROCESSING INPUT FILES")
    print("="*70)

    for file_path in input_files:
        print(f"\n--- Loading: {os.path.basename(file_path)} ---")
        is_ignition = "nonignition" not in file_path.lower()

        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=CHUNK_SIZE, nrows=args.nrows, low_memory=False)):
            chunk["label_ignition"] = 1 if is_ignition else 0
            chunk = ensure_strike_fields(chunk)

            # Downsample non-ignitions before the province lookup
            if not is_ignition and downsample_ratio < 1.0:
                chunk = chunk.sample(frac=downsample_ratio, random_state=RANDOM_SEED + chunk_idx)
            total_input_rows += len(chunk)

            # Assign provinces using OSM boundaries
            chunk_ab, chunk_bc = assign_provinces_to_chunk(chunk, assigner)

            # Count unassigned (not in AB or BC)
            if chunk_ab is None or chunk_bc is None:
                print(f"  ERROR processing chunk {chunk_idx}")
                continue

            other_count = len(chunk) - len(chunk_ab) - len(chunk_bc)

            print(f"  Chunk {chunk_idx}: Alberta={len(chunk_ab):,} | BC={len(chunk_bc):,} | Other={other_count:,}")

            total_ab_rows += len(chunk_ab)
            total_bc_rows += len(chunk_bc)
            total_other_rows += other_count

            # Engineer features and apply final drops for both regions
            chunk_ab_processed = apply_final_drops(engineer_chunk_minimal(chunk_ab)) if len(chunk_ab) > 0 else None
            chunk_bc_processed = apply_final_drops(engineer_chunk_minimal(chunk_bc)) if len(chunk_bc) > 0 else None

            # Set column order on first non-empty chunk
            if col_order is None:
                if chunk_ab_processed is not None and len(chunk_ab_processed) > 0:
                    col_order = chunk_ab_processed.columns.tolist()
                elif chunk_bc_processed is not None and len(chunk_bc_processed) > 0:
                    col_order = chunk_bc_processed.columns.tolist()

            # Write Alberta data
            if chunk_ab_processed is not None and len(chunk_ab_processed) > 0 and col_order:
                chunk_ab_processed = chunk_ab_processed.reindex(columns=col_order)

                # Use header only on first write
                write_header = not ab_file_exists
                chunk_ab_processed.to_csv(OUTPUT_TRAIN_AB, mode='a', index=False, header=write_header)
                ab_file_exists = True

            # Write BC data
            if chunk_bc_processed is not None and len(chunk_bc_processed) > 0 and col_order:
                chunk_bc_processed = chunk_bc_processed.reindex(columns=col_order)

                # Use header only on first write
                write_header = not bc_file_exists
                chunk_bc_processed.to_csv(OUTPUT_TRAIN_BC, mode='a', index=False, header=write_header)
                bc_file_exists = True

    print(f"\n{'='*70}")
    print(f"DATA SPLIT SUMMARY (ALBERTA & BC) - OSM BOUNDARIES")
    print(f"{'='*70}")
    print(f"Total input rows (all files, after non-ignition sampling): {total_input_rows:,}")
    print(f"\nAlberta rows: {total_ab_rows:,} ({(total_ab_rows/total_input_rows*100):.1f}%)")
    print(f"BC rows: {total_bc_rows:,} ({(total_bc_rows/total_input_rows*100):.1f}%)")
    print(f"Other regions/unassigned: {total_other_rows:,} ({(total_other_rows/total_input_rows*100):.1f}%)")
    print(f"\nVerification: {total_ab_rows + total_bc_rows + total_other_rows:,} == {total_input_rows:,}? ", end="")
    if total_ab_rows + total_bc_rows + total_other_rows == total_input_rows:
        print("PASS")
    else:
        print("FAIL - ACCOUNTING ERROR")
    print(f"\nOutput files:")
    print(f"  Alberta: {OUTPUT_TRAIN_AB}")
    print(f"  BC: {OUTPUT_TRAIN_BC}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
