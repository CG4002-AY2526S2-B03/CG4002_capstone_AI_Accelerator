"""
Final training script for Pickleball MTL model.

Run after tuner.py has produced ./artifacts/best_params.json:
    python train.py                          # uses best_params.json automatically
    python train.py --ignore_tuner           # uses hard-coded CONFIG fallback
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, mean_absolute_error, classification_report,
    confusion_matrix, accuracy_score,
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from model import MTLPickleballNet, MTLLoss

HERE         = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS    = os.path.join(HERE, "artifacts")
MODEL_DIR    = os.path.join(ARTIFACTS, "final_model")
PLOT_DIR     = os.path.join(ARTIFACTS, "plots")
SPLITS_PATH  = os.path.join(ARTIFACTS, "splits.npz")
PARAMS_PATH  = os.path.join(ARTIFACTS, "best_params.json")

CLASS_NAMES = ["Drive", "Drop", "Dink", "Lob", "SpeedUp", "HandBattle"]

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#555",      "axes.labelcolor": "#cdd6f4",
    "text.color": "#cdd6f4",       "xtick.color": "#aaa",
    "ytick.color": "#aaa",         "grid.color": "#444",
    "grid.alpha": 0.4,             "legend.facecolor": "#2a2a3c",
    "legend.edgecolor": "#555",    "figure.dpi": 140,
    "savefig.dpi": 180,            "font.size": 9,
})

# Fallback config (used when --ignore_tuner or no best_params.json)
FALLBACK_CONFIG = {
    "input_dim": 6,
    "hidden_dim": 512,
    "num_hidden_layers": 2,
    "regression_output_dim": 6,
    "num_classes": 6,
    "dropout_rate": 0.01,
    "use_batch_norm": False,
    "optimizer": "AdamW",
    "learning_rate_adam": 0.0001,
    "weight_decay": 1e-6,
    "momentum": 0.9,
    "batch_size": 32,
    "num_epochs": 1000,
    "classification_weight": 2.5,
    "regression_weight": 0.35,
}


def load_best_params(params_path: str) -> dict:
    """Load best balanced params from tuner output and merge with defaults."""
    with open(params_path) as f:
        data = json.load(f)
    best = data["best_balanced"]["params"]
    print(f"Loaded tuner params from: {params_path}")
    print(f"  {best}")
    cfg = dict(FALLBACK_CONFIG)
    cfg.update(best)
    return cfg


def build_config(ignore_tuner: bool) -> dict:
    if not ignore_tuner and os.path.exists(PARAMS_PATH):
        return load_best_params(PARAMS_PATH)
    print("Using fallback CONFIG (no tuner params found or --ignore_tuner set).")
    return dict(FALLBACK_CONFIG)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    data = np.load(SPLITS_PATH)
    return {k: data[k] for k in [
        "X_train", "X_val", "X_test",
        "y_reg_train", "y_reg_val", "y_reg_test",
        "y_cls_train", "y_cls_val", "y_cls_test",
    ]}


def compute_class_weights(y_cls, device):
    y = y_cls.astype(np.int64)
    counts = np.bincount(y, minlength=6)
    weights = counts.sum() / (6 * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Curves", fontsize=13, fontweight="bold")
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0,0].plot(ep, history["train_loss"], color="#89b4fa", alpha=0.8, label="Train")
    axes[0,0].plot(ep, history["val_loss"],   color="#f38ba8", alpha=0.8, label="Val")
    axes[0,0].set(ylabel="Loss", xlabel="Epoch", title="Loss"); axes[0,0].legend(); axes[0,0].grid(True)

    best_ep = np.argmax(history["val_f1"]) + 1
    best_f1 = max(history["val_f1"])
    axes[0,1].plot(ep, history["val_f1"], color="#a6e3a1", linewidth=1.5, label="Val Macro F1")
    axes[0,1].axvline(best_ep, color="#f9e2af", linestyle="--", alpha=0.6, label=f"Best @ ep {best_ep}")
    axes[0,1].set(ylabel="Macro F1", xlabel="Epoch", title=f"F1 (best: {best_f1:.4f})")
    axes[0,1].legend(); axes[0,1].grid(True)

    axes[1,0].plot(ep, history["val_mae"], color="#fab387", linewidth=1.5, label="Val MAE")
    axes[1,0].set(ylabel="MAE", xlabel="Epoch", title="Regression MAE"); axes[1,0].legend(); axes[1,0].grid(True)

    axes[1,1].plot(ep, history["lr"], color="#cba6f7", linewidth=1.5, label="LR")
    axes[1,1].set_yscale("log")
    axes[1,1].set(ylabel="LR (log)", xlabel="Epoch", title="Learning Rate Schedule")
    axes[1,1].legend(); axes[1,1].grid(True)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, out_dir, split_name="val"):
    cm = confusion_matrix(y_true, y_pred)
    n  = cm.shape[0]
    names = CLASS_NAMES[:n]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="magma", aspect="auto")
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="#1e1e2e" if cm[i, j] > thresh else "#cdd6f4")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion Matrix ({split_name})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_matrix_{split_name}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_class_f1(y_true, y_pred, out_dir, split_name="val"):
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES[:max(y_true.max(), y_pred.max())+1],
                                   output_dict=True, zero_division=0)
    classes = [c for c in report if c in CLASS_NAMES]
    f1s      = [report[c]["f1-score"] for c in classes]
    supports = [int(report[c]["support"]) for c in classes]
    colors   = ["#a6e3a1" if f >= 0.8 else "#f9e2af" if f >= 0.6 else "#f38ba8" for f in f1s]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(classes, f1s, color=colors, edgecolor="#555", linewidth=0.5)
    for bar, f, s in zip(bars, f1s, supports):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{f:.3f}  (n={s})", va="center", fontsize=8, color="#cdd6f4")
    ax.set_xlim(0, 1.15)
    ax.axvline(0.8, color="#a6e3a1", linestyle="--", alpha=0.3, label="0.8 threshold")
    ax.set(xlabel="F1 Score", title=f"Per-Class F1 ({split_name})")
    ax.legend(fontsize=7); ax.grid(True, axis="x")
    fig.tight_layout()
    path = os.path.join(out_dir, f"per_class_f1_{split_name}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_regression_errors(y_true, y_pred, out_dir, split_name="val"):
    errors = np.abs(y_true - y_pred)
    n_out  = y_true.shape[1]
    names  = ["x_out", "y_out", "z_out", "vx_out", "vy_out", "vz_out"][:n_out]
    mae_per = errors.mean(axis=0)
    colors  = ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7", "#fab387"]

    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(names, mae_per, color=colors[:n_out], edgecolor="#555", linewidth=0.5)
    for i, v in enumerate(mae_per):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=8, color="#cdd6f4")
    ax.set(ylabel="MAE", title=f"Per-Output MAE ({split_name})")
    ax.grid(True, axis="y")

    ax = fig.add_subplot(gs[0, 1])
    for i in range(n_out):
        ax.hist(errors[:, i], bins=40, alpha=0.5, label=names[i], color=colors[i % len(colors)])
    ax.set(xlabel="Absolute Error", ylabel="Count", title=f"Error Distribution ({split_name})")
    ax.legend(fontsize=7); ax.grid(True)

    fig.tight_layout()
    path = os.path.join(out_dir, f"regression_errors_{split_name}.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_split(model, data, split, device):
    X     = torch.FloatTensor(data[f"X_{split}"]).to(device)
    y_reg = data[f"y_reg_{split}"]
    y_cls = data[f"y_cls_{split}"]
    model.eval()
    with torch.no_grad():
        reg_pred, cls_logits = model(X)
        reg_pred  = reg_pred.cpu().numpy()
        cls_pred  = cls_logits.argmax(dim=1).cpu().numpy()
    return {
        "reg_pred": reg_pred, "cls_pred": cls_pred,
        "y_reg":    y_reg,    "y_cls":    y_cls,
        "mae":  mean_absolute_error(y_reg, reg_pred),
        "f1":   f1_score(y_cls, cls_pred, average="macro"),
        "acc":  accuracy_score(y_cls, cls_pred),
    }


def save_report(results, history, config, out_dir):
    report = {
        "config": config,
        "best_epoch": int(np.argmax(history["val_f1"]) + 1),
        "total_epochs_trained": len(history["train_loss"]),
        "val":  {k: float(v) for k, v in results["val"].items()  if k in ("mae", "f1", "acc")},
        "test": {k: float(v) for k, v in results["test"].items() if k in ("mae", "f1", "acc")},
    }
    path = os.path.join(out_dir, "train_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {path}")


# ── Main training loop ─────────────────────────────────────────────────────

def train(ignore_tuner: bool = False):
    device = get_device()
    print(f"Training on device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    cfg = build_config(ignore_tuner)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR,  exist_ok=True)

    print("\nLoading data...")
    data         = load_data()
    class_weights = compute_class_weights(data["y_cls_train"], device)

    # Always derive input_dim from actual data, not config
    cfg["input_dim"] = data["X_train"].shape[1]

    train_ds = TensorDataset(
        torch.FloatTensor(data["X_train"]),
        torch.FloatTensor(data["y_reg_train"]),
        torch.LongTensor(data["y_cls_train"]),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(data["X_val"]),
        torch.FloatTensor(data["y_reg_val"]),
        torch.LongTensor(data["y_cls_val"]),
    )

    batch_size   = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = MTLPickleballNet(
        input_dim=cfg.get("input_dim", 6),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_hidden_layers=int(cfg.get("num_hidden_layers", 3)),
        regression_output_dim=cfg.get("regression_output_dim", 6),
        num_classes=cfg.get("num_classes", 6),
        dropout_rate=float(cfg.get("dropout_rate", 0.01)),
        use_batch_norm=bool(cfg.get("use_batch_norm", False)),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt_name = cfg.get("optimizer", "AdamW")
    lr       = float(cfg.get("learning_rate_adam", cfg.get("learning_rate_sgd", 1e-4)))
    wd       = float(cfg.get("weight_decay", 1e-6))
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    loss_fn = MTLLoss(
        regression_weight=float(cfg.get("regression_weight", 0.35)),
        classification_weight=float(cfg.get("classification_weight", 2.5)),
        class_weights=class_weights,
    )

    num_epochs = int(cfg.get("num_epochs", 1000))
    print(f"\nTraining for {num_epochs} epochs...\n")

    history   = defaultdict(list)
    best_f1   = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for X, y_reg, y_cls in train_loader:
            X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
            optimizer.zero_grad()
            reg_pred, cls_logits = model(X)
            loss, _, _ = loss_fn(reg_pred, y_reg, cls_logits, y_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        rp, rt, cp, ct = [], [], [], []
        with torch.no_grad():
            for X, y_reg, y_cls in val_loader:
                X, y_reg, y_cls = X.to(device), y_reg.to(device), y_cls.to(device)
                reg_pred, cls_logits = model(X)
                v, _, _ = loss_fn(reg_pred, y_reg, cls_logits, y_cls)
                val_loss += v.item()
                rp.append(reg_pred.cpu().numpy()); rt.append(y_reg.cpu().numpy())
                cp.append(cls_logits.argmax(1).cpu().numpy()); ct.append(y_cls.cpu().numpy())
        avg_val  = val_loss / len(val_loader)
        val_mae  = mean_absolute_error(np.concatenate(rt), np.concatenate(rp))
        val_f1   = f1_score(np.concatenate(ct), np.concatenate(cp), average="macro")
        val_acc  = (np.concatenate(ct) == np.concatenate(cp)).mean()
        cur_lr   = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_f1"].append(val_f1)
        history["val_mae"].append(val_mae)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        scheduler.step(val_f1)

        is_best = val_f1 > best_f1
        if is_best:
            best_f1 = val_f1; best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))

        if is_best or (epoch + 1) % 20 == 0 or epoch == 0:
            tag = "★ NEW BEST" if is_best else ""
            print(f"  Epoch {epoch+1:03d} | Loss: {avg_train:.4f}/{avg_val:.4f} | "
                  f"MAE: {val_mae:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.3f} | "
                  f"LR: {cur_lr:.2e} | {time.time()-t0:.1f}s  {tag}")

    print(f"\n{'='*60}")
    print(f"Training Complete — Best F1: {best_f1:.4f} @ epoch {best_epoch}")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), weights_only=True))
    model.eval()

    print("\nEvaluating best model...")
    val_res  = evaluate_split(model, data, "val",  device)
    test_res = evaluate_split(model, data, "test", device)
    print(f"\n  Val  — F1: {val_res['f1']:.4f} | MAE: {val_res['mae']:.4f} | Acc: {val_res['acc']:.4f}")
    print(f"  Test — F1: {test_res['f1']:.4f} | MAE: {test_res['mae']:.4f} | Acc: {test_res['acc']:.4f}")

    for name, res in [("val", val_res), ("test", test_res)]:
        n_cls = max(res["y_cls"].max(), res["cls_pred"].max()) + 1
        print(f"\n  Classification Report ({name}):")
        print(classification_report(res["y_cls"], res["cls_pred"],
                                    target_names=CLASS_NAMES[:n_cls], zero_division=0))

    print("\nGenerating plots...")
    plot_training_curves(history, PLOT_DIR)
    for name, res in [("val", val_res), ("test", test_res)]:
        plot_confusion_matrix(res["y_cls"], res["cls_pred"], PLOT_DIR, name)
        plot_per_class_f1(res["y_cls"], res["cls_pred"], PLOT_DIR, name)
        plot_regression_errors(res["y_reg"], res["reg_pred"], PLOT_DIR, name)

    save_report({"val": val_res, "test": test_res}, history, cfg, MODEL_DIR)

    print("\nExporting to ONNX...")
    dummy = torch.randn(1, cfg.get("input_dim", 6)).to(device)
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input_1"],
        output_names=["dense_3", "dense_4"],
        opset_version=18,
        dynamo=False,
    )
    print(f"Model exported to: {onnx_path}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore_tuner", action="store_true",
                        help="Skip best_params.json and use fallback CONFIG.")
    args = parser.parse_args()
    train(ignore_tuner=args.ignore_tuner)
