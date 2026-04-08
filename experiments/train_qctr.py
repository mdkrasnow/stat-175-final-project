#!/usr/bin/env python3
"""Train the QCTR transition scorer MLP."""

import argparse
import json
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.qctr_model import QCTRDataset, TransitionScorer, evaluate_model
from sklearn.metrics import roc_auc_score


def train(args):
    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "models", "qctr")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    data_path = os.path.join(
        os.path.dirname(__file__), "..", args.data_dir, args.dataset
    )
    data_path = os.path.normpath(data_path)

    train_ds = QCTRDataset(data_path, "train")
    val_ds = QCTRDataset(data_path, "val")
    test_ds = QCTRDataset(data_path, "test")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # ── Initialize model ─────────────────────────────────────────────────
    model = TransitionScorer(dropout=args.dropout).to(device)
    print(f"Parameters: {model.num_parameters:,}")

    # ── Training loop ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_auc = -1.0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, "best_model.pt")

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0
        for features, edge_types, labels in train_loader:
            features = features.to(device)
            edge_types = edge_types.to(device)
            labels = labels.to(device)

            logits = model(features, edge_types)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)

        # ── Validate ─────────────────────────────────────────────────────
        val_metrics = evaluate_model(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"lr={current_lr:.6f} "
            f"({elapsed:.1f}s)"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])

        # ── Early stopping ───────────────────────────────────────────────
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "args": vars(args)},
                best_model_path,
            )
            print(f"  -> Saved best model (val_auc={best_val_auc:.4f})")
        else:
            patience_counter += 1

        scheduler.step(val_metrics["loss"])

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Final evaluation ─────────────────────────────────────────────────
    print("\n--- Final evaluation on test set ---")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(model, test_loader, device)
    print(
        f"Test loss={test_metrics['loss']:.4f}  "
        f"acc={test_metrics['accuracy']:.4f}  "
        f"auc={test_metrics['auc']:.4f}"
    )

    # ── Per-hop evaluation ───────────────────────────────────────────────
    print("\n--- Per-hop evaluation ---")
    meta_path = os.path.join(data_path, "test_metadata.json")
    with open(meta_path) as f:
        test_meta = json.load(f)

    # Collect all predictions on test set
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for features, edge_types, labels in test_loader:
            features = features.to(device)
            edge_types = edge_types.to(device)
            logits = model(features, edge_types)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_probs = torch.sigmoid(all_logits).numpy()
    all_preds = (all_logits > 0).float().numpy()
    labels_np = all_labels.numpy()

    hop_counts = [m["hop_count"] for m in test_meta]
    hop_counts_arr = np.array(hop_counts)

    hop_auc_results = {}
    print(f"{'hop':>4} | {'n_samples':>9} | {'accuracy':>8} | {'auc':>6}")
    print("-" * 40)
    for hop in sorted(set(hop_counts)):
        mask = hop_counts_arr == hop
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float((all_preds[mask] == labels_np[mask]).mean())
        try:
            auc = roc_auc_score(labels_np[mask], all_probs[mask])
        except ValueError:
            auc = float("nan")
        hop_auc_results[hop] = auc
        print(f"{hop:>4} | {n:>9} | {acc:>8.4f} | {auc:>6.4f}")

    # ── Save training curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["val_loss"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Loss")
    axes[1].set_title("Validation Loss")

    axes[2].plot(epochs_range, history["val_auc"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Val AUC")
    axes[2].set_title("Validation AUC")

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"\nTraining curves saved to {curves_path}")

    # ── Checkpoint validation ────────────────────────────────────────────
    print("\n--- Checkpoint validation ---")
    test_auc_pass = test_metrics["auc"] > 0.65
    hop2_auc = hop_auc_results.get(2, 0.0)
    hop2_pass = hop2_auc > 0.60
    model_saved = os.path.exists(best_model_path)

    print(f"Test AUC > 0.65:   {'PASS' if test_auc_pass else 'FAIL'} ({test_metrics['auc']:.4f})")
    print(f"2-hop AUC > 0.60:  {'PASS' if hop2_pass else 'FAIL'} ({hop2_auc:.4f})")
    print(f"Model saved:       {'PASS' if model_saved else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Train QCTR transition scorer")
    parser.add_argument("--dataset", default="prime")
    parser.add_argument("--data-dir", default="data/qctr")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
