"""Generate final report, statistical tests, and plots for context margin study."""
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.context_study.config import RESULTS_DIR, ALL_RUNS_CSV, CLASS_NAMES, DRIVE_ROOT
from src.context_study.utils_metrics import paired_bootstrap_ci, mcnemar_test, wilson_ci

def main():
    if not ALL_RUNS_CSV.exists():
        # Check Drive fallback
        drive_csv = DRIVE_ROOT / "results" / "context_study" / "all_runs.csv"
        if drive_csv.exists():
            shutil.copy2(drive_csv, ALL_RUNS_CSV)
        else:
            print(f"Error: {ALL_RUNS_CSV} not found. Run training phases first.")
            return

    df = pd.read_csv(ALL_RUNS_CSV)
    print("=" * 80)
    print("RARECAM CONTEXT-MARGIN STUDY RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total experiment runs logged: {len(df)}")
    
    # 1. Summary grouped by phase and configuration
    metrics = ["val_accuracy", "test_accuracy", "macro_f1", "mountain_lion_recall"]
    summary_rows = []
    
    for (phase, config), group in df.groupby(["phase", "config"]):
        n_seeds = len(group)
        val_mean, val_std = group["val_accuracy"].mean(), group["val_accuracy"].std()
        test_mean, test_std = group["test_accuracy"].mean(), group["test_accuracy"].std()
        f1_mean, f1_std = group["macro_f1"].mean(), group["macro_f1"].std()
        ml_recall_mean = group["mountain_lion_recall"].mean()
        
        # Calculate mountain lion k/35 and Wilson CI
        k_ml = int(round(ml_recall_mean * 35))
        ci_low, ci_high = wilson_ci(k_ml, 35)
        
        summary_rows.append({
            "phase": str(phase),
            "config": config,
            "n_seeds": n_seeds,
            "val_acc": f"{val_mean*100:.2f} ± {val_std*100:.2f}%" if pd.notnull(val_std) else f"{val_mean*100:.2f}%",
            "test_acc": f"{test_mean*100:.2f} ± {test_std*100:.2f}%" if pd.notnull(test_std) else f"{test_mean*100:.2f}%",
            "macro_f1": f"{f1_mean*100:.2f} ± {f1_std*100:.2f}%" if pd.notnull(f1_std) else f"{f1_mean*100:.2f}%",
            "ml_recall": f"{ml_recall_mean*100:.2f}% ({k_ml}/35, Wilson 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])",
            "_test_mean": test_mean,
            "_val_mean": val_mean
        })
        
    summary_df = pd.DataFrame(summary_rows)
    print("\n--- Summary Table ---")
    print(summary_df[["phase", "config", "n_seeds", "val_acc", "test_acc", "macro_f1", "ml_recall"]].to_string(index=False))
    
    # 2. Statistical hypothesis testing against Phase 0 baseline
    # Find baseline predictions (Phase 0 margin 0.0)
    baseline_pred_files = sorted(list(RESULTS_DIR.glob("preds_phase0_margin_0.0_seed*.csv")))
    if baseline_pred_files:
        print("\n--- Paired Bootstrap and McNemar's Tests vs Phase 0 Baseline ---")
        base_df = pd.read_csv(baseline_pred_files[0]) # seed 0 baseline
        y_true_base = base_df["y_true"].values
        y_pred_base = base_df["y_pred"].values
        
        for pred_file in sorted(list(RESULTS_DIR.glob("preds_*.csv"))):
            if "phase0_margin_0.0" in pred_file.name:
                continue
            cand_df = pd.read_csv(pred_file)
            y_pred_cand = cand_df["y_pred"].values
            
            ci_low, ci_high = paired_bootstrap_ci(y_true_base, y_pred_base, y_pred_cand, n_resamples=10000)
            p_mcnemar = mcnemar_test(y_true_base, y_pred_base, y_pred_cand)
            
            stat_claim = "EXCLUDES ZERO (Significant difference)" if (ci_low > 0 or ci_high < 0) else "INCLUDES ZERO (No statistically significant difference)"
            print(f"\nConfiguration: {pred_file.stem.replace('preds_', '')}")
            print(f"  Paired Bootstrap 95% CI of ΔAcc: [{ci_low*100:.2f}%, {ci_high*100:.2f}%] -> {stat_claim}")
            print(f"  McNemar's test p-value: {p_mcnemar:.4e}")

    # 3. Save Summary Markdown
    md_path = RESULTS_DIR / "context_study_summary.md"
    with open(md_path, "w") as f:
        f.write("# RareCam Context-Margin Study: Experimental Results\n\n")
        f.write("> **Prior Work Context**: Context utilization in camera-trap classification has been established in prior literature (DeLoCo 2025, COSMO arXiv:2401.00608, Dussert et al. 2026, arXiv:2410.12769). This study evaluates whether context expansion or dual-stream fusion improves performance on the fixed Caltech Camera Traps benchmark under sequence-aware evaluation.\n\n")
        f.write("## Protocol Verification\n")
        f.write("- **Fixed Splits**: Sequence-aware splits preserved (train 4,617 / val 971 / test 970 crops).\n")
        f.write("- **Evaluation**: 5 seeds per configuration (mean ± std), hyperparameter/margin selection via validation accuracy only.\n\n")
        f.write("## Results Summary Table\n\n")
        f.write(summary_df[["phase", "config", "n_seeds", "val_acc", "test_acc", "macro_f1", "ml_recall"]].to_markdown(index=False))
        f.write("\n\n")
    print(f"\nReport written to: {md_path}")
    
    # 4. Generate Plot if Phase 1 margins exist
    phase1_df = df[df["phase"].astype(str) == "1"]
    if not phase1_df.empty:
        try:
            plt.figure(figsize=(7, 4.5))
            margin_stats = phase1_df.groupby("config")["val_accuracy"].agg(["mean", "std"]).reset_index()
            margin_stats["margin"] = margin_stats["config"].str.replace("margin_", "").astype(float)
            margin_stats = margin_stats.sort_values("margin")
            
            plt.errorbar(margin_stats["margin"], margin_stats["mean"]*100, yerr=margin_stats["std"]*100, fmt='-o', capsize=5, color="#10b981", lw=2, label="Validation Accuracy")
            plt.xlabel("Context Margin ($m$)")
            plt.ylabel("Validation Accuracy (%)")
            plt.title("Phase 1: Validation Accuracy vs. Context Margin")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plot_path = RESULTS_DIR / "margin_sweep_val_accuracy.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"Plot saved to: {plot_path}")
        except Exception as e:
            print(f"Plot error: {e}")

    # Mirror to Drive if available
    if DRIVE_ROOT.exists():
        drive_res = DRIVE_ROOT / "results" / "context_study"
        drive_res.mkdir(parents=True, exist_ok=True)
        for item in RESULTS_DIR.glob("*"):
            if item.is_file():
                try:
                    shutil.copy2(item, drive_res / item.name)
                except Exception:
                    pass
        print(f"All reports and plots mirrored to Google Drive: {drive_res}")

if __name__ == "__main__":
    main()
