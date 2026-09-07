import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Data from instructions
data = [
    {"Round": 0, "Labels": 944, "Strategy": "Random", "Accuracy": None, "Macro_F1": None, "Mountain_Lion_Recall": None},
    {"Round": 0, "Labels": 944, "Strategy": "Uncertainty", "Accuracy": None, "Macro_F1": None, "Mountain_Lion_Recall": None},
    {"Round": 0, "Labels": 944, "Strategy": "Rarity-aware", "Accuracy": None, "Macro_F1": None, "Mountain_Lion_Recall": None},
    
    {"Round": 1, "Labels": 1094, "Strategy": "Random", "Accuracy": 77.01, "Macro_F1": 74.25, "Mountain_Lion_Recall": 57.14},
    {"Round": 1, "Labels": 1094, "Strategy": "Uncertainty", "Accuracy": 75.98, "Macro_F1": 73.64, "Mountain_Lion_Recall": 54.29},
    {"Round": 1, "Labels": 1094, "Strategy": "Rarity-aware", "Accuracy": 75.05, "Macro_F1": 72.12, "Mountain_Lion_Recall": 57.14},

    {"Round": 2, "Labels": 1244, "Strategy": "Random", "Accuracy": 75.98, "Macro_F1": 71.92, "Mountain_Lion_Recall": 42.86},
    {"Round": 2, "Labels": 1244, "Strategy": "Uncertainty", "Accuracy": 77.63, "Macro_F1": 74.33, "Mountain_Lion_Recall": 51.43},
    {"Round": 2, "Labels": 1244, "Strategy": "Rarity-aware", "Accuracy": 77.94, "Macro_F1": 74.52, "Mountain_Lion_Recall": 45.71},
]

baseline = {"Labels": 4617, "Accuracy": 82.06, "Macro_F1": 81.78, "Mountain_Lion_Recall": 74.29}

df = pd.DataFrame(data)

def plot_metric(metric, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    
    for strategy in df['Strategy'].unique():
        subset = df[df['Strategy'] == strategy]
        plt.plot(subset['Labels'], subset[metric], marker='o', label=strategy)
        
    plt.axhline(y=baseline[metric], color='r', linestyle='--', label='Full Data Baseline')
    
    plt.title(title)
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = Path("results/final") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def main():
    print("Generating plots...")
    plot_metric('Accuracy', 'Accuracy vs Labeling Budget', 'Accuracy (%)', 'accuracy_vs_budget.png')
    plot_metric('Macro_F1', 'Macro-F1 vs Labeling Budget', 'Macro-F1 (%)', 'macro_f1_vs_budget.png')
    plot_metric('Mountain_Lion_Recall', 'Mountain Lion Recall vs Labeling Budget', 'Recall (%)', 'mountain_lion_recall_vs_budget.png')
    
    # Save CSV
    output_csv = Path("results/final/active_learning_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv.parent}")

if __name__ == "__main__":
    main()
