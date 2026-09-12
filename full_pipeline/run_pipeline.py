"""
run_pipeline.py
================
Runs the ENTIRE project end-to-end, one step per week of the plan:

    Step 1  Data acquisition           -> src/generate_synthetic_data.py
    Step 2  Alignment & features       -> src/align_and_engineer.py
    Step 3-4 Calibration models        -> src/calibration_models.py (used by evaluate_models.py)
    Step 5  Evaluation & selection     -> src/evaluate_models.py
    Step 6  Exposure gap & placement   -> src/exposure_gap.py, src/placement.py
    (extra) Visualization              -> src/visualize.py

After this script finishes, everything the FastAPI service (api/main.py) and
the Streamlit dashboard (dashboard/app.py) need is sitting in
outputs/results/*.json and data/processed/.

Usage:
    python run_pipeline.py                 # full run (regenerates synthetic data)
    python run_pipeline.py --skip-generate # reuse whatever is already in data/raw/
                                            # (use this once you drop in REAL data)
"""

import argparse
import time

from src import generate_synthetic_data, align_and_engineer, evaluate_models, exposure_gap, placement, visualize


def main(skip_generate: bool = False) -> None:
    t0 = time.time()

    print("\n=== STEP 1: DATA ACQUISITION ===")
    if skip_generate:
        print("[run_pipeline] --skip-generate set: using existing files in data/raw/")
    else:
        generate_synthetic_data.generate_all()

    print("\n=== STEP 2: ALIGNMENT & FEATURE ENGINEERING ===")
    align_and_engineer.run()

    print("\n=== STEP 3-5: MODEL TRAINING, EVALUATION & SELECTION ===")
    eval_results = evaluate_models.run()

    print("\n=== STEP 6a: POPULATION-WEIGHTED EXPOSURE GAP ===")
    gap_results = exposure_gap.run()

    print("\n=== STEP 6b: GREEDY SENSOR PLACEMENT ===")
    placement_results = placement.run()

    print("\n=== EXTRA: VISUALIZATION ===")
    visualize.run()

    elapsed = time.time() - t0
    print(f"\n=== PIPELINE COMPLETE in {elapsed:.1f}s ===")
    print(f"Selected model: {eval_results['selected_model']}")
    print("City summary (population-weighted exposure vs. official monitor average):")
    for city, r in gap_results.items():
        print(
            f"  {city:12s} pop-weighted={r['population_weighted_exposure_ugm3']:6.1f}  "
            f"official={r['official_monitor_average_ugm3']:6.1f}  "
            f"gap={r['exposure_gap_ugm3']:+6.1f} ug/m3 ({r['exposure_gap_pct']:+.1f}%)"
        )
    print("\nResults written to outputs/results/*.json, maps to outputs/plots/*.png")
    print("Next steps:")
    print("  - Serve results:   uvicorn api.main:app --reload")
    print("  - View dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the air-quality sensor audit pipeline end-to-end.")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing data/raw/ files instead of regenerating synthetic data.")
    args = parser.parse_args()
    main(skip_generate=args.skip_generate)
