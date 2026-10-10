"""Generate final report and plots."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.context_study.utils_metrics import paired_bootstrap_ci, mcnemar_test, wilson_ci
from src.context_study.config import *

def main():
    # 1. Read ALL_RUNS_CSV
    # 2. Compute means, std devs
    # 3. Perform statistical tests (McNemar and Paired Bootstrap) vs Phase 0 baseline
    # 4. Generate Markdown/LaTeX tables
    # 5. Generate plots (Accuracy vs Margin, Ablation bar chart)
    pass

if __name__ == "__main__":
    main()
