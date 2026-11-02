#!/usr/bin/env python3
"""Runs the pipeline scripts in src/ in order.

    data/combined_*.csv
      -> step1_feature_engineering.py   -> data/processed/train_alberta_osm.csv (+ train_bc_osm.csv)
      -> step2_diagnostics.py           (checks on the processed Alberta file)
      -> step3_train_model.py           -> outputs/training_artifacts/ (optional)
      -> step4_inference.py             -> outputs/predictions_AB.csv
      -> step5_postprocess_for_ui.py    -> outputs/ShowCaseUIInput.csv
      -> step6_map_app.py               (Streamlit map; upload ShowCaseUIInput.csv)

Training is not part of --all because inference uses the existing model in
model/. Use --train-quick to check that training runs, or --train for the
full Optuna + Monte Carlo run.

Examples:
    python run_pipeline.py --all
    python run_pipeline.py --all --no-ui --nonignition-frac 0.01
    python run_pipeline.py --feature-engineering --inputs all
    python run_pipeline.py --inference --inference-csv some_file.csv
    python run_pipeline.py --train-quick
    python run_pipeline.py --ui
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "src"

sys.path.insert(0, str(SCRIPTS_DIR))
from common import TRAIN_AB_CSV, PREDICTIONS_CSV, UI_INPUT_CSV  # noqa: E402

# Save plots to file instead of opening windows
CHILD_ENV = {**os.environ, "MPLBACKEND": "Agg"}


def run_python_script(script_name: str, *args: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    command = [sys.executable, str(script_path), *args]
    print("\n" + "=" * 90)
    print(f"Running: {' '.join(command)}")
    print("=" * 90, flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), env=CHILD_ENV)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_streamlit_app(app_name: str) -> None:
    app_path = SCRIPTS_DIR / app_name
    if not app_path.exists():
        raise FileNotFoundError(f"App not found: {app_path}")

    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("\n" + "=" * 90)
    print(f"Launching Streamlit app: {' '.join(command)}")
    if UI_INPUT_CSV.exists():
        print(f"Upload this file in the app: {UI_INPUT_CSV}")
    print("=" * 90, flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), env=CHILD_ENV)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightning prediction pipeline from the project root."
    )
    steps = parser.add_argument_group("steps")
    steps.add_argument("--all", action="store_true",
                       help="Steps 1, 2, 4, 5 and 6: feature engineering to map UI.")
    steps.add_argument("--feature-engineering", action="store_true", help="Step 1: feature engineering")
    steps.add_argument("--diagnostic", action="store_true", help="Step 2: data diagnostics")
    steps.add_argument("--train", action="store_true", help="Step 3: full model training (slow)")
    steps.add_argument("--train-quick", action="store_true",
                       help="Step 3 on a small subsample, to check training runs (about a minute)")
    steps.add_argument("--inference", action="store_true", help="Step 4: inference")
    steps.add_argument("--postprocess", action="store_true", help="Step 5: post-processing for the UI")
    steps.add_argument("--ui", action="store_true", help="Step 6: launch the map UI")

    opts = parser.add_argument_group("options")
    opts.add_argument("--no-ui", action="store_true", help="With --all, stop before launching the UI.")
    opts.add_argument("--inputs", choices=["14_to_18", "all"], default="14_to_18",
                      help="Raw files for feature engineering (default: 14_to_18).")
    opts.add_argument("--nonignition-frac", type=float, default=0.05,
                      help="Fraction of non-ignition strikes kept in feature engineering (default: 0.05).")
    opts.add_argument("--nrows", type=int, default=None,
                      help="Only read the first N rows of each raw file.")
    opts.add_argument("--diagnostic-csv", default=str(TRAIN_AB_CSV),
                      help="CSV for the diagnostic step (default: processed Alberta file).")
    opts.add_argument("--inference-csv", default=str(TRAIN_AB_CSV),
                      help="CSV for the inference step (default: processed Alberta file).")
    opts.add_argument("--train-csv", default=str(TRAIN_AB_CSV),
                      help="CSV for the training step (default: processed Alberta file).")
    args = parser.parse_args()

    selected = [args.feature_engineering, args.diagnostic, args.inference, args.postprocess,
                args.ui, args.train, args.train_quick]
    if not args.all and not any(selected):
        parser.print_help()
        return

    run_fe = args.all or args.feature_engineering
    run_diag = args.all or args.diagnostic
    run_inf = args.all or args.inference
    run_post = args.all or args.postprocess
    run_ui = (args.all and not args.no_ui) or args.ui

    if run_fe:
        fe_args = ["--inputs", args.inputs, "--nonignition-frac", str(args.nonignition_frac)]
        if args.nrows:
            fe_args += ["--nrows", str(args.nrows)]
        run_python_script("step1_feature_engineering.py", *fe_args)

    if run_diag:
        run_python_script("step2_diagnostics.py", args.diagnostic_csv)

    if args.train_quick:
        run_python_script("step3_train_model.py", "--quick", "--train-csv", args.train_csv)

    if args.train:
        run_python_script("step3_train_model.py", "--train-csv", args.train_csv)

    if run_inf:
        run_python_script("step4_inference.py", args.inference_csv)

    if run_post:
        run_python_script("step5_postprocess_for_ui.py", "--input", str(PREDICTIONS_CSV),
                          "--output", str(UI_INPUT_CSV))

    if run_ui:
        run_streamlit_app("step6_map_app.py")
    elif args.all:
        print(f"\nPipeline finished. UI input: {UI_INPUT_CSV}")
        print("To launch the map, run: python run_pipeline.py --ui")


if __name__ == "__main__":
    main()
