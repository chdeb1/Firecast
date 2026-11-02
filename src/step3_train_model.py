import os
import argparse
import joblib
import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, brier_score_loss
)
from xgboost import XGBClassifier

from common import TRAIN_AB_CSV, OUTPUT_DIR as PIPELINE_OUTPUT_DIR, encode_fuel_type

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_DATA_CSV = str(TRAIN_AB_CSV)
# Kept separate from model/ so training never overwrites the existing model
OUTPUT_DIR = str(PIPELINE_OUTPUT_DIR / "training_artifacts")
RANDOM_SEED = 42
N_REPEATS = 20
OPTUNA_TRIALS = 20
TEST_SIZE = 0.30
VAL_FRACTION = 0.50
LABEL_COL = "label_ignition"

# Maximum trees for Optuna trials and Monte Carlo repeats (early stopping applies)
OPTUNA_N_ESTIMATORS = 400
MC_N_ESTIMATORS = 5000

# --quick: runs every step on a small subsample
QUICK_N_REPEATS = 1
QUICK_OPTUNA_TRIALS = 2
QUICK_N_ESTIMATORS = 50
QUICK_MAX_ROWS = 50_000

# Strictly exclude metadata, spatial fingerprints, and removed density features
DROP_COLS = [
    "fire_index", "strike_id", "source_file", "dataset_type", "geometry",
    "ignition_lat", "ignition_lon", "ignition_date",
    "strike_datetime", "strike_date", "strike_lat", "strike_lon",
    "distance_km", "cloud_or_ground", "peak_signal_kA",
    "Peak Signal (kiloAmps)", "Latitude", "Longitude",
    "dry_lightning_index", "storm_strike_density"
]

# ----------------------------
# HELPERS
# ----------------------------
def find_best_f1_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]), float(f1s[best_idx])

def precision_at_recall(y_true, y_prob, target_recall):
    """Find precision at a specific recall level"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Find closest recall to target
    idx = np.argmin(np.abs(recalls - target_recall))
    return float(precisions[idx]), float(thresholds[idx])

def compute_calibration_metrics(y_true, y_prob):
    """Compute calibration curve and Brier score"""
    # Brier score: mean squared error of probabilities
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve: how well predicted probabilities match observed frequencies
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')

    return brier, prob_true, prob_pred

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the XGBoost ignition model")
    parser.add_argument("--train-csv", default=TRAIN_DATA_CSV, help="Feature-engineered training CSV")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Where artifacts and the model are saved")
    parser.add_argument("--quick", action="store_true",
                        help="Quick check: 1 repeat, 2 Optuna trials, 50 trees, at most 50k rows")
    args = parser.parse_args()

    output_dir = args.output_dir
    n_repeats, optuna_trials = N_REPEATS, OPTUNA_TRIALS
    optuna_n_estimators, mc_n_estimators = OPTUNA_N_ESTIMATORS, MC_N_ESTIMATORS
    if args.quick:
        n_repeats, optuna_trials = QUICK_N_REPEATS, QUICK_OPTUNA_TRIALS
        optuna_n_estimators = mc_n_estimators = QUICK_N_ESTIMATORS
        output_dir = os.path.join(output_dir, "quick")
        print(f"QUICK MODE: {n_repeats} repeat, {optuna_trials} trials, {QUICK_N_ESTIMATORS} trees, <= {QUICK_MAX_ROWS:,} rows")

    os.makedirs(output_dir, exist_ok=True)

    # 1) Load Processed Data
    print(f"Loading simplified data from {args.train_csv}...")
    df = pd.read_csv(args.train_csv, low_memory=False)

    # Clean label and ensure numeric
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()

    # Keep only valid labels (0 and 1)
    df = df[df['label_ignition'].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if args.quick and len(df) > QUICK_MAX_ROWS:
        # Keep every ignition, fill the rest with non-ignitions
        pos = df[df[LABEL_COL] == 1]
        neg = df[df[LABEL_COL] == 0]
        n_neg = max(QUICK_MAX_ROWS - len(pos), 0)
        df = pd.concat([pos, neg.sample(n=min(n_neg, len(neg)), random_state=RANDOM_SEED)])
        print(f"  Subsampled to {len(df):,} rows ({len(pos):,} ignitions)")

    y = df[LABEL_COL].to_numpy()

    # Dynamic Lag Drop (keep lag_0 and avg_24h only as per research)
    lags_to_drop = [col for col in df.columns if any(suffix in col for suffix in ["_lag_1", "_lag_2", "_lag_24"])]
    final_drop_list = DROP_COLS + lags_to_drop
    # Keep dropped/metadata columns for later re-attachment
    meta_cols = [c for c in final_drop_list if c in df.columns and c != LABEL_COL]
    meta_df = df[meta_cols].copy()

    # Feature matrix preparation
    X = df.drop(columns=[c for c in final_drop_list if c in df.columns], errors="ignore")
    X = X.drop(columns=[LABEL_COL], errors="ignore")

    if 'fuel_type_code' in X.columns:
        X['fuel_type_code'] = encode_fuel_type(X['fuel_type_code']).to_numpy()

    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()

    # Leakage diagnostic
    print("\nRunning Leakage Diagnostic (single-feature AUC)...")
    for col in X.columns:
        xcol = X[col].to_numpy()
        # Skip constant columns
        if np.nanstd(xcol) < 1e-12:
            continue
        try:
            auc = roc_auc_score(y, xcol)
            # Flag strong separation either direction
            if max(auc, 1-auc) > 0.85:
                print(f"!! ALERT !! Feature '{col}' single-feature AUROC={auc:.4f} (strong signal; check leakage)")
        except ValueError:
            # e.g., only one class present after filtering, etc.
            continue

    # 3) Bayesian Optimization (Optuna)
    # Optimizes AUPRC, which suits heavily imbalanced data
    def objective(trial):

        # New split per trial to avoid tuning to a single split
        seed = RANDOM_SEED + trial.number
        xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        spw = (yt == 0).sum() / (yt == 1).sum()

        params = {
            'n_estimators': optuna_n_estimators,
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'scale_pos_weight': spw,
            'tree_method': 'hist',
            'n_jobs': -1
        }

        model = XGBClassifier(**params, objective="binary:logistic", eval_metric="aucpr", early_stopping_rounds=25)

        model.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
        preds = model.predict_proba(xv, iteration_range=(0, model.best_iteration + 1))[:, 1]
        actual_auprc = average_precision_score(yv, preds)

        return actual_auprc

    print("\nStarting Bayesian Optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=optuna_trials)

    # 4) Monte Carlo Loop with Best Params
    all_results = []
    best_model = None
    best_val_auprc = -1

    for i in range(n_repeats):
        print(f"\n===== REPEAT {i+1}/{n_repeats} =====")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED + i, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_FRACTION, random_state=RANDOM_SEED + i, stratify=y_temp)

        params = dict(study.best_params)
        params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

        # Early stopping on validation AUPRC
        xgb = XGBClassifier(**params, n_estimators=mc_n_estimators, tree_method='hist', eval_metric="aucpr", early_stopping_rounds=100, n_jobs=-1, objective="binary:logistic",)
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        best_iter = xgb.best_iteration
        test_prob = xgb.predict_proba(X_test, iteration_range=(0, best_iter + 1))[:, 1]
        val_prob  = xgb.predict_proba(X_val,  iteration_range=(0, best_iter + 1))[:, 1]

        best_thr, _ = find_best_f1_threshold(y_val, val_prob)
        test_pred = (test_prob >= best_thr).astype(int)

        val_auprc = average_precision_score(y_val, val_prob)

        test_roc_auc = roc_auc_score(y_test, test_prob)

        # Precision at specific recall levels
        prec_at_50_recall, thr_50 = precision_at_recall(y_test, test_prob, target_recall=0.50)
        prec_at_70_recall, thr_70 = precision_at_recall(y_test, test_prob, target_recall=0.70)
        prec_at_90_recall, thr_90 = precision_at_recall(y_test, test_prob, target_recall=0.90)

        # Calibration metrics
        test_brier, _, _ = compute_calibration_metrics(y_test, test_prob)

        res = {
            "repeat": i + 1,
            "best_iter": int(best_iter),
            "VAL_AUPRC": float(val_auprc),
            "ROC_AUC": float(test_roc_auc),
            "AUPRC": average_precision_score(y_test, test_prob),
            "F1": f1_score(y_test, test_pred),
            "Brier_Score": float(test_brier),
            "Prec_at_50_Recall": float(prec_at_50_recall),
            "Prec_at_70_Recall": float(prec_at_70_recall),
            "Prec_at_90_Recall": float(prec_at_90_recall),
            "Threshold_50R": float(thr_50),
            "Threshold_70R": float(thr_70),
            "Threshold_90R": float(thr_90),
            "F1_Threshold": best_thr
        }
        all_results.append(res)
        # Select the best model on validation AUPRC
        if val_auprc > best_val_auprc:
          best_val_auprc = val_auprc
          best_model = xgb

          best_X_test = X_test.copy()
          best_meta_test = meta_df.loc[X_test.index].copy()

          best_y_test = pd.Series(y_test, index=X_test.index, name=LABEL_COL)
          best_test_prob = pd.Series(test_prob, index=X_test.index, name="prob")
          best_test_pred = pd.Series(test_pred, index=X_test.index, name="pred")
          best_threshold = best_thr
          best_test_roc_auc = test_roc_auc
          best_brier = test_brier


    # 5) SHAP Interpretation
    print("\nGenerating SHAP summary...")
    explainer = shap.TreeExplainer(best_model)
    X_shap = best_X_test.sample(min(500, len(best_X_test)), random_state=RANDOM_SEED)
    shap_values = explainer.shap_values(X_shap)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_ALBERTA.png"))

    # 6) Generate Calibration Curve
    print("\nGenerating calibration curve...")
    brier, prob_true, prob_pred = compute_calibration_metrics(best_y_test.values, best_test_prob.values)

    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('True Frequency of Positive Class', fontsize=12)
    plt.title(f'Calibration Curve (Brier Score: {brier:.4f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150)
    plt.close()

    # 7) Generate Precision-Recall Curve with Recall Levels Marked
    print("\nGenerating precision-recall curve with recall levels...")
    precisions, recalls, thresholds = precision_recall_curve(best_y_test.values, best_test_prob.values)

    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')

    # Mark the specific recall levels
    for target_recall in [0.50, 0.70, 0.90]:
        idx = np.argmin(np.abs(recalls - target_recall))
        plt.plot(recalls[idx], precisions[idx], 'o', markersize=10, label=f'Recall={target_recall:.0%}')

    plt.xlabel('Recall (% of real fires caught)', fontsize=12)
    plt.ylabel('Precision (% of alarms that are real)', fontsize=12)
    plt.title('Precision-Recall Curve with Operating Points', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve_with_levels.png"), dpi=150)
    plt.close()

    # 8) Save metrics and feature importance
    summary = pd.DataFrame(all_results)
    summary.to_csv(os.path.join(output_dir, "mc_metrics_v2.csv"), index=False)

    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    importance.to_csv(os.path.join(output_dir, "feature_importanceAB.csv"), index=False)
    best_test_full = pd.concat([
        best_meta_test.reset_index(drop=True),
        best_X_test.reset_index(drop=True),
        best_y_test.reset_index(drop=True),
        best_test_prob.reset_index(drop=True),
        best_test_pred.reset_index(drop=True),
    ], axis=1)

    best_test_full.to_csv(os.path.join(output_dir, "best_test_data_Mar16_AB.csv"), index=False)

    joblib.dump(best_model, os.path.join(output_dir, "xgb_mac_v2_Mar16_AB.joblib"))

    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
    print("="*70)
    print(f"\nMean ROC-AUC: {summary['ROC_AUC'].mean():.4f} (+/- {summary['ROC_AUC'].std():.4f})")
    print(f"Mean AUPRC: {summary['AUPRC'].mean():.4f} (+/- {summary['AUPRC'].std():.4f})")
    print(f"Mean F1 Score: {summary['F1'].mean():.4f} (+/- {summary['F1'].std():.4f})")
    print(f"Mean Brier Score: {summary['Brier_Score'].mean():.4f} (+/- {summary['Brier_Score'].std():.4f})")

    print(f"\n--- PRECISION AT DIFFERENT RECALL LEVELS ---")
    print(f"Precision @ 50% Recall: {summary['Prec_at_50_Recall'].mean():.4f} (threshold: {summary['Threshold_50R'].mean():.3f})")
    print(f"Precision @ 70% Recall: {summary['Prec_at_70_Recall'].mean():.4f} (threshold: {summary['Threshold_70R'].mean():.3f})")
    print(f"Precision @ 90% Recall: {summary['Prec_at_90_Recall'].mean():.4f} (threshold: {summary['Threshold_90R'].mean():.3f})")

    print(f"\n--- BEST MODEL SELECTED ---")
    print(f"Best Model ROC-AUC: {best_test_roc_auc:.4f}")
    print(f"Best Model Brier Score: {best_brier:.4f}")

    print(f"\nTop Features: {importance['feature'].head(5).tolist()}")
    print("\n" + "="*70)
    print(f"Training Complete. Files saved in {output_dir}:")
    print("  - mc_metrics_v2.csv (all metrics for all repeats)")
    print("  - calibration_curve.png")
    print("  - precision_recall_curve_with_levels.png")
    print("  - shap_summary_ALBERTA.png")
    print("  - feature_importanceAB.csv")
    print("="*70)

if __name__ == "__main__":
    main()
