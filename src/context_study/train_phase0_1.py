"""Training script for Phase 0 (tight crops) and Phase 1 (context-margin sweep)."""
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.context_study.config import *
from src.context_study.dataset import DynamicContextDataset
from src.context_study.models import SingleStreamResNet
from src.context_study.utils_metrics import calculate_all_metrics, log_result
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def train_one_configuration(margin, seed, is_phase_0=False):
    phase = "0" if is_phase_0 else "1"
    config_name = f"margin_{margin}"
    chk_path = RESULTS_DIR / f"phase{phase}_{config_name}_seed{seed}.pth"
    preds_path = RESULTS_DIR / f"preds_phase{phase}_{config_name}_seed{seed}.csv"
    
    # Resumability check: skip if already executed and logged
    if chk_path.exists() and preds_path.exists() and ALL_RUNS_CSV.exists():
        try:
            df_runs = pd.read_csv(ALL_RUNS_CSV)
            existing = df_runs[(df_runs["phase"].astype(str) == str(phase)) & 
                               (df_runs["config"] == config_name) & 
                               (df_runs["seed"] == seed)]
            if not existing.empty:
                print(f"[RESUME] Skipping already completed run: Phase {phase} | {config_name} | Seed {seed}")
                return
        except Exception as e:
            print(f"Warning checking resume status: {e}")

    set_seed(seed)
    train_transform, eval_transform = get_transforms()
    
    train_ds = DynamicContextDataset(SPLITS_DIR / "train.csv", IMAGES_DIR, DETECTIONS_JSON, CLASS_NAMES, margin=margin, transform=train_transform)
    val_ds = DynamicContextDataset(SPLITS_DIR / "val.csv", IMAGES_DIR, DETECTIONS_JSON, CLASS_NAMES, margin=margin, transform=eval_transform)
    test_ds = DynamicContextDataset(SPLITS_DIR / "test.csv", IMAGES_DIR, DETECTIONS_JSON, CLASS_NAMES, margin=margin, transform=eval_transform)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nw = 2 if device == "cuda" else 0
    pm = (device == "cuda")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, pin_memory=pm)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pm)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pm)
    
    model = SingleStreamResNet(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_val_acc = -1
    best_state = None
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                logits = model(images)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(labels.numpy())
                
        val_acc, _, _ = calculate_all_metrics(np.array(val_labels), np.array(val_preds), CLASS_NAMES)
        print(f"Phase {phase} | {config_name} | Seed {seed} | Epoch {epoch+1:02d}/{MAX_EPOCHS} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            
    # Evaluate on test with best model
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            test_preds.extend(logits.argmax(1).cpu().numpy())
            test_labels.extend(labels.numpy())
            
    test_acc, test_macro, per_class = calculate_all_metrics(np.array(test_labels), np.array(test_preds), CLASS_NAMES)
    
    # Save checkpoint
    torch.save(best_state, chk_path)
    
    # Save per-sample test predictions
    df_preds = pd.DataFrame({
        "y_true": test_labels,
        "y_pred": test_preds
    })
    df_preds.to_csv(preds_path, index=False)
    
    # Mirror checkpoint and predictions to Drive if mounted
    drive_root = Path("/content/gdrive/MyDrive/RareCam")
    if drive_root.exists():
        drive_res = drive_root / "results" / "context_study"
        drive_res.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(chk_path, drive_res / chk_path.name)
            shutil.copy2(preds_path, drive_res / preds_path.name)
        except Exception as e:
            print(f"Warning: Could not sync checkpoint to Drive: {e}")
            
    log_result(ALL_RUNS_CSV, phase, config_name, seed, best_val_acc, test_acc, test_macro, per_class)
    print(f"=== COMPLETED: Phase {phase} | Config: {config_name} | Seed: {seed} | Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.phase == 0:
        margins = [0.0]
    else:
        margins = [0.0, 0.25, 0.5, 1.0]
        
    for margin in margins:
        for seed in SEEDS:
            train_one_configuration(margin, seed, is_phase_0=(args.phase == 0))
