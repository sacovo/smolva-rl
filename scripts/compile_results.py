#!/usr/bin/env python3
import os
import json
import pandas as pd
from pathlib import Path

# Combinations used in run_array_task.py
CHECKPOINTS = [
    ("150k", "outputs/libero_recap_150000"),
    ("250k", "outputs/libero_recap_250000"),
    ("350k (final)", "outputs/libero/recap_model/migrated")
]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
CFG_SCALES = [0.0, 1.0, 1.5, 2.0]

def find_eval_info(job_name):
    eval_dir = Path("outputs/eval")
    if not eval_dir.exists():
        return None
    for path in eval_dir.glob(f"**/[0-9]*-[0-9]*-[0-9]*_{job_name}/eval_info.json"):
        if path.exists():
            return path
    return None

def main():
    results = []
    missing = []
    
    print("Scanning outputs/eval for completed evaluation runs...")
    
    for cp_name, cp_path in CHECKPOINTS:
        for suite in SUITES:
            for cfg in CFG_SCALES:
                job_name = f"recap_{cp_name}_{suite}_cfg_{cfg}".replace(" ", "_").replace("(", "").replace(")", "")
                info_path = find_eval_info(job_name)
                
                if info_path and info_path.exists():
                    try:
                        with open(info_path, 'r') as f:
                            data = json.load(f)
                            success_rate = data.get("overall", {}).get("pc_success", 0.0)
                            n_episodes = data.get("overall", {}).get("n_episodes", 0)
                            results.append({
                                "Checkpoint": cp_name,
                                "Suite": suite,
                                "CFG Weight": float(cfg),
                                "Success Rate (%)": success_rate,
                                "Episodes": n_episodes
                            })
                    except Exception as e:
                        print(f"Error reading {info_path}: {e}")
                else:
                    missing.append(f"Checkpoint: {cp_name} | Suite: {suite} | CFG: {cfg} (Job Name: {job_name})")
                    
    if missing:
        print(f"\nWarning: {len(missing)} out of 48 evaluation runs are missing/incomplete:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more.")
    else:
        print("\nAll 48 evaluation runs located successfully!")
        
    if not results:
        print("No evaluation results found to compile.")
        return
        
    df = pd.DataFrame(results)
    pivoted = df.pivot_table(
        index=["Checkpoint", "CFG Weight"],
        columns="Suite",
        values="Success Rate (%)"
    )
    
    existing_suites = [s for s in SUITES if s in pivoted.columns]
    pivoted = pivoted[existing_suites]
    pivoted["Average"] = pivoted.mean(axis=1)
    
    print("\n" + "="*90)
    print("RECAP EVALUATION SWEEP SUMMARY")
    print("="*90)
    print(pivoted.round(2).to_string())
    print("="*90)
    
    os.makedirs("outputs/eval", exist_ok=True)
    df.to_csv("outputs/eval/sweep_summary.csv", index=False)
    pivoted.to_csv("outputs/eval/sweep_summary_pivoted.csv")
    print("\nResults compiled and saved to:")
    print("  - outputs/eval/sweep_summary.csv (flat data)")
    print("  - outputs/eval/sweep_summary_pivoted.csv (pivoted matrix)")
    
    try:
        excel_path = "outputs/eval/sweep_summary_report.xlsx"
        pivoted.to_excel(excel_path)
        print(f"  - {excel_path} (Excel matrix report)")
    except Exception as e:
        print(f"Note: Could not generate Excel report: {e}")

if __name__ == "__main__":
    main()
