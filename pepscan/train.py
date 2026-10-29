"""Training and evaluation utilities for PepScan 2.0."""

import os
import copy
import random
import time
import json
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    matthews_corrcoef, f1_score, confusion_matrix,
)

from pepscan.tasks import TASK_NAMES, NUM_TASKS, DL_HYPERPARAMS

warnings.filterwarnings('ignore')


# ============================================================
# Seed & Utilities
# ============================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_bce_loss(logits, labels, mask, task_weights=None):
    """Masked binary cross-entropy loss for multi-task training.

    Args:
        logits: (B, num_tasks) raw logits.
        labels: (B, num_tasks) target labels (0 or 1).
        mask: (B, num_tasks) 1.0 for observed, 0.0 for missing.
        task_weights: Optional (num_tasks,) per-task loss weights.

    Returns:
        Scalar loss.
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels,
                                              reduction='none')
    if task_weights is not None:
        loss = loss * task_weights.to(logits.device).unsqueeze(0)
    masked = loss * mask
    return masked.sum() / mask.sum().clamp(min=1)


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute classification metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Threshold for binary prediction.

    Returns:
        Dict with AUROC, AUPRC, ACC, MCC, F1, Sensitivity, Specificity.
    """
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float('nan')
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = float('nan')

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sen = spe = 0.0

    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'ACC': float(acc),
        'MCC': float(mcc),
        'F1': float(f1),
        'Sensitivity': float(sen),
        'Specificity': float(spe),
    }


# ============================================================
# Model C Training (Single-task MLP)
# ============================================================

def train_model_c(task, train_X, train_y, val_X, val_y,
                  seed=42, max_epochs=50, batch_size=64, lr=1e-3,
                  patience=8, device=None):
    """Train a single-task MLP (Model C).

    Args:
        task: Task name (for logging).
        train_X: Training ESM-2 embeddings (N, 640).
        train_y: Training labels (N,).
        val_X: Validation embeddings.
        val_y: Validation labels.
        seed: Random seed.
        max_epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        device: torch device.

    Returns:
        Trained ModelC instance.
    """
    from pepscan.models import ModelC

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    train_ds = TensorDataset(torch.FloatTensor(train_X),
                             torch.FloatTensor(train_y))
    val_ds = TensorDataset(torch.FloatTensor(val_X),
                           torch.FloatTensor(val_y))

    # BN fix: ensure batch >= 2
    safe_batch = min(batch_size, max(2, len(train_ds) // 2))
    train_loader = DataLoader(train_ds, batch_size=safe_batch,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(2, safe_batch * 2),
                            shuffle=False)

    model = ModelC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    best_loss, best_state, wait = float('inf'), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += F.binary_cross_entropy_with_logits(
                    model(xb), yb).item() * len(xb)
        vl /= len(val_ds)
        scheduler.step(vl)

        if vl < best_loss:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model


# ============================================================
# Model D/E Training (Multi-task)
# ============================================================

def train_multitask(model, train_data, val_data, task_weights=None,
                    max_epochs=50, batch_size=64, lr=1e-3, patience=8,
                    grad_clip=1.0, model_name="model", save_dir=None,
                    device=None):
    """Train a multi-task model (Model D or E).

    Uses masked BCE loss and cosine annealing LR schedule.
    Supports incremental checkpoint saving.

    Args:
        model: ModelD or ModelE instance.
        train_data: Multi-task training data (from MultiTaskDataset).
        val_data: Multi-task validation data.
        task_weights: Optional (num_tasks,) per-task loss weights.
        max_epochs: Maximum training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        patience: Early stopping patience.
        grad_clip: Gradient clipping max norm.
        model_name: Name for checkpoint files.
        save_dir: Directory for saving checkpoints. If None, no saving.
        device: torch device.

    Returns:
        Tuple of (trained model, best_val_loss, training_logs).
    """
    from pepscan.data import MultiTaskDataset

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = MultiTaskDataset(train_data['embeddings'],
                                train_data['labels'])
    val_ds = MultiTaskDataset(val_data['embeddings'],
                              val_data['labels'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if task_weights is None:
        task_weights = torch.ones(NUM_TASKS)
    task_weights_dev = task_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    epoch_logs = []

    # Checkpoint paths
    best_ckpt = None
    last_ckpt = None
    log_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt = os.path.join(save_dir, f"{model_name}_best.pt")
        last_ckpt = os.path.join(save_dir, f"{model_name}_last.pt")
        log_path = os.path.join(save_dir, f"{model_name}_train_log.json")

    for epoch in range(max_epochs):
        model.train()
        total_loss, n_bat = 0, 0
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            logits = model(xb)
            loss = masked_bce_loss(logits, yb, mb, task_weights_dev)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_bat += 1
        scheduler.step()
        avg_loss = total_loss / n_bat

        # Validate
        model.eval()
        with torch.no_grad():
            vlogits, vlabels, vmasks = [], [], []
            for xb, yb, mb in DataLoader(val_ds, batch_size=256):
                vlogits.append(model(xb.to(device)).cpu())
                vlabels.append(yb)
                vmasks.append(mb)
            vlogits = torch.cat(vlogits).to(device)
            vlabels = torch.cat(vlabels).to(device)
            vmasks = torch.cat(vmasks).to(device)
            val_loss = masked_bce_loss(vlogits, vlabels, vmasks,
                                       task_weights_dev)

        # Per-task val AUROC
        val_aurocs = []
        for i in range(NUM_TASKS):
            mi = vmasks[:, i].bool()
            y = vlabels[mi, i].cpu().numpy()
            p = torch.sigmoid(vlogits[mi, i]).cpu().numpy()
            if mi.sum() > 10 and y.sum() > 0 and (1 - y).sum() > 0:
                val_aurocs.append(roc_auc_score(y, p))
        mean_auc = np.mean(val_aurocs) if val_aurocs else 0

        log_entry = {
            'epoch': epoch + 1, 'train_loss': avg_loss,
            'val_loss': val_loss.item(), 'val_auroc': mean_auc,
        }
        epoch_logs.append(log_entry)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  [{model_name}] Epoch {epoch+1:>3}/{max_epochs} "
                  f"loss={avg_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_AUROC={mean_auc:.4f}")

        # Save checkpoints
        if save_dir:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, last_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
            if save_dir:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                }, best_ckpt)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  [{model_name}] Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final save
    if save_dir:
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"{model_name}_final.pt"))
        if log_path:
            with open(log_path, 'w') as f:
                json.dump(epoch_logs, f, indent=2)

    print(f"  [{model_name}] Done. Best val_loss={best_val_loss:.4f}")
    return model, best_val_loss, epoch_logs


# ============================================================
# Evaluation
# ============================================================

def evaluate_single_task(model, test_X, test_y, device=None):
    """Evaluate a single-task model (Model C) on test data.

    Args:
        model: Trained ModelC instance.
        test_X: Test embeddings (N, 640).
        test_y: Test labels (N,).
        device: torch device.

    Returns:
        Dict of metrics from compute_metrics().
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(test_X).to(device)
        probs = torch.sigmoid(model(x)).cpu().numpy()

    return compute_metrics(test_y, probs)


def evaluate_multitask(model, test_data, device=None):
    """Evaluate a multi-task model (Model D or E) on test data.

    Args:
        model: Trained ModelD or ModelE instance.
        test_data: Multi-task test data with 'embeddings' and 'labels'.
        device: torch device.

    Returns:
        Dict mapping task name → metrics dict.
    """
    from pepscan.data import MultiTaskDataset

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MultiTaskDataset(test_data['embeddings'], test_data['labels'])
    loader = DataLoader(ds, batch_size=256)

    model.eval()
    all_logits, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for xb, yb, mb in loader:
            all_logits.append(model(xb.to(device)).cpu())
            all_labels.append(yb)
            all_masks.append(mb)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_masks = torch.cat(all_masks)

    results = {}
    for i, task in enumerate(TASK_NAMES):
        mi = all_masks[:, i].bool()
        if mi.sum() == 0:
            continue
        y = all_labels[mi, i].numpy()
        p = torch.sigmoid(all_logits[mi, i]).numpy()
        if y.sum() > 0 and (1 - y).sum() > 0:
            results[task] = compute_metrics(y, p)
        else:
            results[task] = {k: float('nan')
                             for k in compute_metrics([0], [0.5])}

    return results
