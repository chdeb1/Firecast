import argparse
import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

from common import (
    MODEL_PATH, TRAIN_AB_CSV, OUTPUT_DIR, PREDICTIONS_CSV, UNCERTAIN_PREDICTIONS_CSV,
    METRICS_CSV, UNCERTAIN_METRICS_CSV, LOGIC_PLOTS_DIR, encode_fuel_type,
)

OUT_PATH = PREDICTIONS_CSV
UNCERTAIN_OUT_PATH = UNCERTAIN_PREDICTIONS_CSV

# Columns that should NOT be used as model inputs (metadata / labels / etc.)
DROP_COLS = [
    "fire_index", "strike_id", "source_file", "dataset_type", "geometry",
    "ignition_lat", "ignition_lon", "ignition_date",
    "strike_datetime", "strike_date", "strike_lat", "strike_lon",
    "distance_km", "cloud_or_ground", "peak_signal_kA",
    "Peak Signal (kiloAmps)", "Latitude", "Longitude",
    "dry_lightning_index", "storm_strike_density",
    "label_ignition", "prob", "pred", "ignition_probability", "prediction"
]

parser = argparse.ArgumentParser(description="Score lightning strikes with the trained model")
parser.add_argument("csv_path", nargs="?", default=str(TRAIN_AB_CSV),
                    help="CSV to score (default: Alberta output of feature engineering)")
parser.add_argument("--model", default=str(MODEL_PATH), help="Path to the .joblib model")
parser.add_argument("--fuel-already-encoded", action="store_true",
                    help="Input fuel_type_code already holds model codes (e.g. best_test_data CSV from step 3)")
cli = parser.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load input CSV
csv_path = cli.csv_path
print(f"Scoring {csv_path} with {cli.model}")
df = pd.read_csv(csv_path, low_memory=False)

# ---- Add ground_truth logic
if "dataset_type" in df.columns:
    # normalize strings
    dt = df["dataset_type"].astype(str).str.strip().str.lower()
    mapping = {"ignition": 1, "nonignition": 0}
    df["ground_truth"] = dt.map(mapping)
elif "label_ignition" in df.columns:
    # Use the column created by the Feature Engineering script
    df["ground_truth"] = df["label_ignition"]
else:
    print("WARNING: No label_ignition or dataset_type found. Metrics will be skipped.")
    df["ground_truth"] = pd.NA

# ---- Load model
model = joblib.load(cli.model)

# ---- Build feature matrix (do NOT pass metadata to the model)
feature_order = getattr(model, "feature_names_in_", None)

if feature_order is not None:
    # Use only model features, in the right order
    X = df.reindex(columns=list(feature_order), fill_value=0)
else:
    # Fallback: drop obvious non-features
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

# Raw fuel IDs to training codes
if "fuel_type_code" in X.columns and not cli.fuel_already_encoded:
    X["fuel_type_code"] = encode_fuel_type(df["fuel_type_code"]).to_numpy()

# ---- Predict

# THRESHOLD = 0.3308551609516144   # optimally found through training
THRESHOLD = 0.50  # default

df["ignition_probability"] = model.predict_proba(X)[:, 1]
df["prediction"] = (df["ignition_probability"] >= THRESHOLD).astype(int)

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, accuracy_score
)

# ---- Metrics (only on rows with known ground truth)
eval_df = df.dropna(subset=["ground_truth"]).copy()
if len(eval_df) == 0:
    print("No ground_truth values available; skipping metrics.")
else:
    y_true = eval_df["ground_truth"].astype(int).to_numpy()
    y_prob = eval_df["ignition_probability"].to_numpy()
    y_pred = eval_df["prediction"].astype(int).to_numpy()

    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = pd.DataFrame([{
        "n_eval": len(eval_df),
        "positive_rate": float(y_true.mean()),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "precision": float(prec),
        "recall": float(rec),
        "F1": float(f1),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }])

    metrics_path = METRICS_CSV
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved metrics to {metrics_path}")

# ----------------------------
# Metrics for UNCERTAIN range only
# ----------------------------

LOW = 0.05
HIGH = 0.95

uncertain_df = df[
    (df["ignition_probability"] >= LOW) &
    (df["ignition_probability"] <= HIGH)
].dropna(subset=["ground_truth"]).copy()

if len(uncertain_df) == 0:
    print("No uncertain rows with ground_truth available.")
else:
    y_true = uncertain_df["ground_truth"].astype(int).to_numpy()
    y_prob = uncertain_df["ignition_probability"].to_numpy()
    y_pred = uncertain_df["prediction"].astype(int).to_numpy()

    # Handle single-class edge case safely
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else float("nan")
    auprc = average_precision_score(y_true, y_prob)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    ratio = pos / neg if neg > 0 else float("inf")

    uncertain_metrics = pd.DataFrame([{
        "range": f"{LOW}-{HIGH}",
        "n_uncertain": len(uncertain_df),
        "positive_rate": float(y_true.mean()),
        "pos_neg_ratio": float(ratio),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "F1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }])

    uncertain_metrics.to_csv(UNCERTAIN_METRICS_CSV, index=False)

    print("\nUncertain-range metrics:")
    print(uncertain_metrics.to_string(index=False))
    print(f"Saved {UNCERTAIN_METRICS_CSV}")

# ---- Make sure the columns the UI needs exist
for col in ["strike_lat", "strike_lon", "dataset_type"]:
    if col not in df.columns:
        df[col] = pd.NA

df.to_csv(OUT_PATH, index=False)

# ---- Save "uncertain" subset

df[(df["ignition_probability"] >= LOW) &
   (df["ignition_probability"] <= HIGH)
].to_csv(UNCERTAIN_OUT_PATH, index=False)



print(f"Saved to {OUT_PATH}")

# --- 1. Native XGBoost Tree Visualization ---
def save_model_blueprint(model):
    print("Generating model blueprint (Tree 0)...")
    # Set figure size large enough to read the nodes
    fig, ax = plt.subplots(figsize=(30, 15))

    # Plotting the first tree (index 0)
    # 'rankdir=LR' makes the tree flow left-to-right
    xgb.plot_tree(model, num_trees=0, ax=ax, rankdir='LR')

    plt.title("XGBoost Model Blueprint - Tree 0")
    plt.savefig(os.path.join(LOGIC_PLOTS_DIR, "0_model_blueprint_tree.png"), bbox_inches='tight', dpi=300)
    plt.close()

def save_individual_paths(model, X, df):
    import shap
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute SHAP only for the rows being plotted
    df['dist_to_thresh'] = (df['ignition_probability'] - 0.5).abs()
    uncertain_indices = df.nsmallest(3, 'dist_to_thresh').index.tolist()
    idx_ign = df['ignition_probability'].idxmax()
    idx_non = df['ignition_probability'].idxmin()

    selected_indices = [idx_ign, idx_non] + uncertain_indices
    labels = ["Obvious_Ignition", "Obvious_Non_Ignition", "Uncertain_1", "Uncertain_2", "Uncertain_3"]
    X_sel = X.loc[selected_indices]

    explainer = shap.TreeExplainer(model)
    shap_val_output = explainer.shap_values(X_sel)

    # Extract the "Ignition" slice correctly
    if isinstance(shap_val_output, list):
        current_shap_values = shap_val_output[1]
        current_expected_value = explainer.expected_value[1]
    elif len(shap_val_output.shape) == 3:
        current_shap_values = shap_val_output[:, :, 1]
        current_expected_value = explainer.expected_value[1]
    else:
        current_shap_values = shap_val_output
        current_expected_value = explainer.expected_value

    for i, idx in enumerate(selected_indices):
        prob = df.loc[idx, 'ignition_probability']

        # Increase figure size for better vertical spacing
        fig = plt.figure(figsize=(10, 10))

        # FEATURE_DISPLAY_RANGE: Only show top 15 features to avoid clutter
        shap.decision_plot(
            current_expected_value,
            current_shap_values[i],
            X_sel.iloc[i],
            feature_names=X.columns.tolist(),
            link="logit",
            feature_display_range=slice(-1, -16, -1), # Show top 15
            show=False
        )

        # FORCE X-AXIS SCALE: Ensures 0% and 100% are always visible
        plt.xlim(0, 1)
        plt.title(f"{labels[i]} | Predicted Risk: {prob:.2%}")
        plt.tight_layout()

        plt.savefig(os.path.join(LOGIC_PLOTS_DIR, f"path_{i+1}_{labels[i]}.png"), bbox_inches='tight')
        plt.close(fig)
        print(f"Generated clean path for {labels[i]}")

# Execute both
os.makedirs(LOGIC_PLOTS_DIR, exist_ok=True)
try:
    save_model_blueprint(model)
except Exception as e:
    # xgb.plot_tree requires Graphviz
    print(f"Skipping model blueprint plot ({type(e).__name__}: {e})")
save_individual_paths(model, X, df)
print(f"Saved uncertain rows to {UNCERTAIN_OUT_PATH}")
