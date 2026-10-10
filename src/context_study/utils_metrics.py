"""Statistical metrics and utilities for context study."""
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from collections import defaultdict

def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    return (center - margin) / denominator, (center + margin) / denominator

def paired_bootstrap_ci(y_true, y_pred1, y_pred2, n_resamples=10000, confidence=0.95):
    """Paired bootstrap 95% CI for the difference in accuracy between two models."""
    acc1 = (y_true == y_pred1).astype(float)
    acc2 = (y_true == y_pred2).astype(float)
    diffs = acc2 - acc1
    n = len(diffs)
    
    rng = np.random.RandomState(42)
    bootstrap_diffs = []
    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        bootstrap_diffs.append(np.mean(diffs[indices]))
        
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)
    return lower, upper

def mcnemar_test(y_true, y_pred1, y_pred2):
    """McNemar's test for paired nominal data."""
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    if b + c == 0:
        return 1.0 # p-value 1 if exactly identical
    
    statistic = (abs(b - c) - 1)**2 / (b + c)
    p_value = stats.chi2.sf(statistic, 1)
    return p_value

def calculate_all_metrics(y_true, y_pred, class_names):
    """Calculates accuracy, macro F1, and per-class recall."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    per_class = {}
    for i, name in enumerate(class_names):
        mask = (y_true == i)
        if np.sum(mask) > 0:
            recall = np.sum(y_pred[mask] == i) / np.sum(mask)
        else:
            recall = 0.0
        per_class[f"{name}_recall"] = recall
        
    return acc, macro_f1, per_class

def log_result(csv_path, phase, config_name, seed, val_acc, test_acc, macro_f1, per_class_recalls, commit_hash=""):
    """Appends a row to the all_runs.csv file safely."""
    row = {
        "phase": phase,
        "config": config_name,
        "seed": seed,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "commit_hash": commit_hash
    }
    row.update(per_class_recalls)
    
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode='w', header=True, index=False)
