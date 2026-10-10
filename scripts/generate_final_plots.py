import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np

def generate_line_plot(df, metric, title, ylabel, out_path, baseline_val=None):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='labeled_samples', y=metric, hue='strategy', marker='o', linewidth=2)
    
    if baseline_val is not None:
        plt.axhline(y=baseline_val, color='r', linestyle='--', label='Full-Data Baseline')
        
    plt.title(title, fontsize=14)
    plt.xlabel('Labeling Budget (Samples)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_confusion_matrix(cm_path, title, out_path):
    if not cm_path.exists():
        return
    cm_df = pd.read_csv(cm_path, index_col=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    results_dir = Path("results/active_learning")
    exp_dir = Path("results/experiments")
    out_dir = Path("results/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Budgets mapped from rounds (seed=944, R1=1094, R2=1244, R3=1394)
    budget_map = {0: 944, 1: 1094, 2: 1244, 3: 1394}
    
    # 1. Gather all historical active learning results (original)
    data = []
    strategies = ["random", "uncertainty", "rarity_weighted"]
    for strategy in strategies:
        for r in [1, 2, 3]:
            path = results_dir / strategy / f"round_{r}" / "summary.csv"
            if path.exists():
                df = pd.read_csv(path)
                row = df.iloc[0].to_dict()
                # If old format didn't have 'strategy', add it
                row['strategy'] = strategy
                row['round'] = r
                row['labeled_samples'] = budget_map[r]
                # Normalize column names if necessary
                acc = row.get('accuracy', row.get('Accuracy'))
                macro = row.get('macro_f1', row.get('Macro_F1'))
                ml_recall = row.get('mountain_lion_recall', row.get('Mountain_Lion_Recall'))
                
                data.append({
                    'strategy': strategy,
                    'labeled_samples': budget_map[r],
                    'accuracy': acc,
                    'macro_f1': macro,
                    'mountain_lion_recall': ml_recall
                })
                
    if not data:
        print("No historical CSVs found to plot budgets.")
    else:
        df_history = pd.DataFrame(data)
        
        # Baselines from the paper draft
        baseline_acc = 0.8206
        baseline_macro = 0.8178
        baseline_ml_recall = 0.7429
        
        generate_line_plot(df_history, 'accuracy', 'Test Accuracy vs Labeling Budget', 'Accuracy', out_dir / 'accuracy_vs_budget.png', baseline_acc)
        generate_line_plot(df_history, 'macro_f1', 'Macro-F1 vs Labeling Budget', 'Macro-F1', out_dir / 'macro_f1_vs_budget.png', baseline_macro)
        generate_line_plot(df_history, 'mountain_lion_recall', 'Mountain Lion Recall vs Labeling Budget', 'Recall', out_dir / 'ml_recall_vs_budget.png', baseline_ml_recall)
        print("Generated line plots.")

    # 2. Gather Exp B (Transfer Learning) results
    exp_data = []
    for strategy in strategies:
        path = exp_dir / "expB" / strategy / "round_3" / "summary.csv"
        if path.exists():
            df = pd.read_csv(path)
            row = df.iloc[0].to_dict()
            exp_data.append({
                'Strategy': strategy,
                'Type': 'Exp B (Warmup+Finetune)',
                'Accuracy': row.get('accuracy', 0),
                'Macro_F1': row.get('macro_f1', 0),
                'Mountain_Lion_Recall': row.get('mountain_lion_recall', 0)
            })
            
            # Confusion Matrix
            cm_path = exp_dir / "expB" / strategy / "round_3" / "confusion_matrix.csv"
            plot_confusion_matrix(cm_path, f"Confusion Matrix: Exp B - {strategy.capitalize()} (Round 3)", out_dir / f"cm_expB_{strategy}.png")
            
        # Get Original Round 3 for comparison
        orig_path = results_dir / strategy / "round_3" / "summary.csv"
        if orig_path.exists():
            df = pd.read_csv(orig_path)
            row = df.iloc[0].to_dict()
            exp_data.append({
                'Strategy': strategy,
                'Type': 'Original Pipeline',
                'Accuracy': row.get('accuracy', row.get('Accuracy', 0)),
                'Macro_F1': row.get('macro_f1', row.get('Macro_F1', 0)),
                'Mountain_Lion_Recall': row.get('mountain_lion_recall', row.get('Mountain_Lion_Recall', 0))
            })

    if exp_data:
        df_comp = pd.DataFrame(exp_data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_comp, x='Strategy', y='Macro_F1', hue='Type')
        plt.title('Round 3 Macro-F1: Original vs. Experiment B', fontsize=14)
        plt.ylabel('Macro-F1', fontsize=12)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / 'round3_comparison_macro_f1.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_comp, x='Strategy', y='Mountain_Lion_Recall', hue='Type')
        plt.title('Round 3 Mountain Lion Recall: Original vs. Experiment B', fontsize=14)
        plt.ylabel('Recall', fontsize=12)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / 'round3_comparison_ml_recall.png', dpi=300)
        plt.close()
        print("Generated bar comparisons and confusion matrices.")
        
    print(f"All plots saved to {out_dir.absolute()}")

if __name__ == "__main__":
    main()
