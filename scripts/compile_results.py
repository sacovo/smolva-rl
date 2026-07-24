#!/usr/bin/env python3
from pathlib import Path


from lerobot_policy_smolvla_rl.analyze.eval_results import collect_eval_results

def main():
    print("Scanning outputs/eval for completed evaluation runs...")
    eval_dir = Path("outputs/eval")
    df = collect_eval_results(eval_dir)
    
    if df.empty:
        print("No evaluation results found to compile.")
        return
        
    print(f"Found {len(df)} completed evaluation runs.")
    
    # Pivot matches compile_results.py behavior: Checkpoint, CFG Weight, Suite
    if "checkpoint" in df.columns and "suite" in df.columns and "cfg" in df.columns:
        pivoted = df.pivot_table(
            index=["checkpoint", "cfg"],
            columns="suite",
            values="pc_success"
        )
        
        # Sort column orders matching SUITES if present
        SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        existing_suites = [s for s in SUITES if s in pivoted.columns]
        pivoted = pivoted[existing_suites]
        pivoted["Average"] = pivoted.mean(axis=1)
        
        print("\n" + "="*90)
        print("RECAP EVALUATION SWEEP SUMMARY")
        print("="*90)
        print(pivoted.round(2).to_string())
        print("="*90)
        
        pivoted.to_csv("outputs/eval/sweep_summary_pivoted.csv")
        
    df.to_csv("outputs/eval/sweep_summary.csv", index=False)
    print("\nResults compiled and saved to:")
    print("  - outputs/eval/sweep_summary.csv (flat data)")
    print("  - outputs/eval/sweep_summary_pivoted.csv (pivoted matrix)")

if __name__ == "__main__":
    main()
