import argparse
import pandas as pd
import numpy as np
import sys

from common import PREDICTIONS_CSV, UI_INPUT_CSV, OSM_CACHE_DIR, OSMProvinceAssigner

# ----------------------------
# CONFIG
# ----------------------------
INPUT_CSV = str(PREDICTIONS_CSV)
OUTPUT_CSV = str(UI_INPUT_CSV)

# Downsampling target ratio (ignitions : non-ignitions)
TARGET_RATIO = 10  # 1:10

# ----------------------------
# MAIN
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter predictions to Alberta and downsample for the UI")
    parser.add_argument("--input", default=INPUT_CSV, help="Predictions CSV from step 4")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Where to write the UI input CSV")
    args = parser.parse_args()
    input_csv, output_csv = args.input, args.output

    print("\n" + "="*70)
    print("DOWNSAMPLING & FILTERING PREDICTIONS")
    print("="*70)

    # Load predictions
    print(f"\nLoading predictions from {input_csv}...")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    print(f"\nInitial class distribution:")
    print(f"  Ignitions (1): {(df['label_ignition']==1).sum():,}")
    print(f"  Non-ignitions (0): {(df['label_ignition']==0).sum():,}")

    # Initialize OSM assigner
    print("\n" + "-"*70)
    print("INITIALIZING OPENSTREETMAP ASSIGNER")
    print("-"*70)
    assigner = OSMProvinceAssigner(OSM_CACHE_DIR)

    # Filter to Alberta
    print("\nFiltering to Alberta using OSM boundaries...")

    if "strike_lat" not in df.columns or "strike_lon" not in df.columns:
        print("ERROR: Missing strike_lat or strike_lon columns")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    df["in_alberta"] = assigner.is_in_alberta(df["strike_lat"], df["strike_lon"])
    df_alberta = df[df["in_alberta"]].drop(columns=["in_alberta"]).copy()

    print(f"  Filtered to Alberta: {len(df_alberta):,} rows ({100*len(df_alberta)/len(df):.1f}% of original)")

    print(f"\nAlberta class distribution (before downsampling):")
    ab_ign = (df_alberta['label_ignition']==1).sum()
    ab_non_ign = (df_alberta['label_ignition']==0).sum()
    print(f"  Ignitions (1): {ab_ign:,}")
    print(f"  Non-ignitions (0): {ab_non_ign:,}")
    print(f"  Current ratio: 1:{ab_non_ign/ab_ign if ab_ign > 0 else 0:.1f}")

    # Downsample to target ratio
    print(f"\nDownsampling to 1:{TARGET_RATIO} ratio...")

    ignitions = df_alberta[df_alberta['label_ignition']==1].copy()
    non_ignitions = df_alberta[df_alberta['label_ignition']==0].copy()

    # Calculate how many non-ignitions to keep
    target_non_ign_count = len(ignitions) * TARGET_RATIO

    if len(non_ignitions) > target_non_ign_count:
        # Downsample non-ignitions
        non_ignitions_sampled = non_ignitions.sample(
            n=int(target_non_ign_count),
            random_state=42
        )
        downsampled_count = int(target_non_ign_count)
        removed_count = len(non_ignitions) - downsampled_count
    else:
        # Keep all non-ignitions (ratio is already less extreme)
        non_ignitions_sampled = non_ignitions.copy()
        downsampled_count = len(non_ignitions)
        removed_count = 0

    # Combine
    df_final = pd.concat([ignitions, non_ignitions_sampled], ignore_index=True)

    # Shuffle to randomize order
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  Removed {removed_count:,} non-ignition samples")
    print(f"  Shuffled rows randomly")

    print(f"\nFinal class distribution (after downsampling):")
    final_ign = (df_final['label_ignition']==1).sum()
    final_non_ign = (df_final['label_ignition']==0).sum()
    print(f"  Ignitions (1): {final_ign:,}")
    print(f"  Non-ignitions (0): {final_non_ign:,}")
    print(f"  Final ratio: 1:{final_non_ign/final_ign:.1f}")

    # Save
    print(f"\nSaving filtered and downsampled predictions to {output_csv}...")
    df_final.to_csv(output_csv, index=False)
    print(f"  Saved {len(df_final):,} rows")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original rows: {len(df):,}")
    print(f"After Alberta filter: {len(df_alberta):,} ({100*len(df_alberta)/len(df):.1f}%)")
    print(f"After downsampling: {len(df_final):,}")
    print(f"Non-ignitions removed: {removed_count:,}")
    print(f"Final ratio: 1:{final_non_ign/final_ign:.1f}")
    print(f"\nOutput file: {output_csv}")
    print("="*70)

if __name__ == "__main__":
    main()
