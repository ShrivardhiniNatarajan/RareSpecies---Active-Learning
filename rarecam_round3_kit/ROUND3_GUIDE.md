# RareCam Round 3 completion kit

This kit completes the **existing three-strategy Round 3** (Random, Uncertainty, Rarity-aware). It does not implement the separate untested balanced `0.7 U + 0.3 R` idea.

Round 3 protocol:
- Starting labeled data for each strategy: `labeled_round_2.csv` (1,244 rows).
- Per-round acquisition budget: 150 crops, with at most one selected crop per `seq_id`.
- After acquisition: `labeled_round_3.csv` must contain 1,394 rows.
- Use `*_round_2.pth` only to select Round-3 samples.
- Train Round-3 models using the same protocol as Rounds 1–2: ImageNet-pretrained ResNet-50, 10 epochs, AdamW, learning rate `1e-4`, batch size 32; save the best validation checkpoint.
- Evaluate each Round-3 model on the unchanged `data/splits/test.csv` (970 rows).
- Do not use hidden labels from the unlabeled pool in the selection score. Labels are retained in the CSV only to simulate the human oracle after selection; the inference-only dataset reads only `crop_file`.

## 1. Runtime and repo

Use a GPU runtime for training. Colab free resources are dynamic; if no GPU is available, use a Kaggle GPU notebook if your account has accelerator quota available. Do not run all three training jobs on CPU unless you accept a much longer runtime.

Mount Drive in Colab first:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

Change directory to the repo root (`/content/RareSpecies---Active-Learning`). If the Colab runtime reset and the repo is gone, clone your own GitHub origin URL, then `cd` into it.

## 2. Add the files

Copy this kit's `src/active_learning/acquire_round3.py`, `src/active_learning/train_round.py`, and `src/active_learning/evaluate_round.py` into the same paths in the repository. Copy `scripts/restore_round3_assets.py` and `scripts/backup_round3_assets.py` into `scripts/`.

Review changes, especially if your local scripts have additional project-specific edits. Then commit/push the source code and pull it into the compute notebook:

```powershell
git diff --check
git add src/active_learning/acquire_round3.py src/active_learning/train_round.py src/active_learning/evaluate_round.py scripts/restore_round3_assets.py scripts/backup_round3_assets.py
git commit -m "feat: complete round 3 active learning pipeline"
git push
```

## 3. Restore Round-2 artifacts

From the repo root, after Drive is mounted:

```python
!python scripts/restore_round3_assets.py
```

The restore script expects these Drive files:

- `RareCam/megadetector/cropped_images.zip`
- `RareCam/classification/train.csv`, `val.csv`, `test.csv`
- `RareCam/active_learning/round_2/{strategy}_selected_round_2.csv`, `{strategy}_remaining_round_2.csv`, `{strategy}_labeled_round_2.csv`
- `RareCam/active_learning/models/{strategy}_round_2.pth`

The script fails with the exact missing paths if the backup is incomplete. It will not silently recreate Round-2 data or models.

## 4. Check GPU

```python
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

If `CUDA available` is `False`, switch to a GPU runtime if possible. Colab doesn't guarantee GPU availability; use Kaggle if needed and if quota is available.

## 5. Acquire Round 3 samples

Run each strategy separately:

```python
!python -m src.active_learning.acquire_round3 --strategy random
!python -m src.active_learning.acquire_round3 --strategy uncertainty
!python -m src.active_learning.acquire_round3 --strategy rarity_weighted
```

For each strategy, verify:

- `selected_round_3.csv`: 150 rows and 150 unique `seq_id`
- `labeled_round_3.csv`: 1,394 rows
- no selected Round-3 `seq_id` overlaps `labeled_round_2.csv`
- selected sequence IDs are absent from `remaining_round_3.csv`

The acquisition script prints the selected ground-truth class distribution only as a post-selection simulation diagnostic. That label column is not used for scoring. Rarity weights come only from `labeled_round_2.csv`.

Back up selection artifacts before training:

```python
!python scripts/backup_round3_assets.py
```

(The backup script copies any currently existing Round-3 artifacts; run it again after models/results are created.)

## 6. Train Round-3 models

Run one at a time and back up the checkpoint after each:

```python
!python -m src.active_learning.train_round --strategy random --round 3
!python -m src.active_learning.train_round --strategy uncertainty --round 3
!python -m src.active_learning.train_round --strategy rarity_weighted --round 3
```

Expected checkpoints:

- `models/active_learning/random_round_3.pth`
- `models/active_learning/uncertainty_round_3.pth`
- `models/active_learning/rarity_weighted_round_3.pth`

To back up between jobs, run `!python scripts/backup_round3_assets.py` after each model. If some result files don't exist yet, the script reports `SKIP` and continues.

## 7. Evaluate on the unchanged test set

```python
!python -m src.active_learning.evaluate_round --strategy random --round 3
!python -m src.active_learning.evaluate_round --strategy uncertainty --round 3
!python -m src.active_learning.evaluate_round --strategy rarity_weighted --round 3
```

Each run saves `summary.csv`, `classification_report.csv`, `per_species_metrics.csv`, `per_species_recall.csv`, and `confusion_matrix.csv` under `results/active_learning/<strategy>/round_3/`.

After evaluations, back up everything:

```python
!python scripts/backup_round3_assets.py
```

## 8. Round-3 comparison table

Run this Colab cell:

```python
from pathlib import Path
import pandas as pd

rows = []
for strategy in ['random', 'uncertainty', 'rarity_weighted']:
    path = Path('results/active_learning') / strategy / 'round_3' / 'summary.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing evaluation summary: {path}')
    row = pd.read_csv(path).iloc[0].to_dict()
    rows.append(row)

comparison = pd.DataFrame(rows)
print(comparison.to_string(index=False))
out = Path('results/active_learning/round_3_comparison.csv')
out.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(out, index=False)
print('Saved:', out)
```

Report the actual results even if rarity-aware does not win. Do not invent Round-3 numbers and do not claim the untested balanced variant was evaluated.
