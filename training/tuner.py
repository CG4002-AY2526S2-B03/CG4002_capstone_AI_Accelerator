"""
Multi-Objective Optuna Tuner for Pickleball MTL Model.
Objectives: Normalized MAE (minimize) and -F1 (minimize).

Run after prepare_dataset.py:
    python tuner.py --n_trials 5000

Best trial params are saved to ./artifacts/best_params.json.
"""

import argparse
import json
import os
import gc
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, mean_absolute_error
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore", message=".*pin_memory.*")

from model import MTLPickleballNet, MTLLoss

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
RANDOM_SEED = 24538074598 % (2**32)


def set_seed(seed: int = RANDOM_SEED):
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(y_cls: np.ndarray) -> torch.Tensor:
    y = y_cls.astype(np.int64)
    num_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if total > 0:
        weights[nonzero] = total / (num_classes * counts[nonzero])
    else:
        weights[:] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def load_data(data_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Run prepare_dataset.py first."
        )
    data = np.load(data_path)
    return {k: data[k] for k in ["X_train", "X_val", "y_reg_train", "y_reg_val", "y_cls_train", "y_cls_val"]}


def create_dataloaders(data: Dict[str, np.ndarray], batch_size: int) -> Tuple[DataLoader, DataLoader]:
    use_pin_memory = (DEVICE.type == "cuda")

    def make_ds(X, y_reg, y_cls):
        return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_reg), torch.LongTensor(y_cls))

    train_loader = DataLoader(make_ds(data["X_train"], data["y_reg_train"], data["y_cls_train"]),
                              batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_pin_memory)
    val_loader   = DataLoader(make_ds(data["X_val"],   data["y_reg_val"],   data["y_cls_val"]),
                              batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_pin_memory)
    return train_loader, val_loader


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer, loss_fn: MTLLoss) -> float:
    model.train()
    total = 0.0
    nb = DEVICE.type == "cuda"
    for X, y_reg, y_cls in loader:
        X, y_reg, y_cls = X.to(DEVICE, non_blocking=nb), y_reg.to(DEVICE, non_blocking=nb), y_cls.to(DEVICE, non_blocking=nb)
        optimizer.zero_grad()
        reg_pred, cls_logits = model(X)
        loss, _, _ = loss_fn(reg_pred, y_reg, cls_logits, y_cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: MTLLoss) -> Dict[str, float]:
    model.eval()
    reg_preds, reg_tgts, cls_preds, cls_tgts = [], [], [], []
    nb = DEVICE.type == "cuda"
    with torch.no_grad():
        for X, y_reg, y_cls in loader:
            X, y_reg, y_cls = X.to(DEVICE, non_blocking=nb), y_reg.to(DEVICE, non_blocking=nb), y_cls.to(DEVICE, non_blocking=nb)
            reg_pred, cls_logits = model(X)
            reg_preds.append(reg_pred.cpu().numpy())
            reg_tgts.append(y_reg.cpu().numpy())
            cls_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
            cls_tgts.append(y_cls.cpu().numpy())
    return {
        "mae":      mean_absolute_error(np.concatenate(reg_tgts), np.concatenate(reg_preds)),
        "f1_score": f1_score(np.concatenate(cls_tgts), np.concatenate(cls_preds), average="macro"),
    }


# Singleton so dataloaders are only created once per process
_RESOURCES: Dict = {}

def get_resources(data: Dict[str, np.ndarray], batch_size: int) -> Dict:
    if not _RESOURCES:
        class_weights = compute_class_weights(data["y_cls_train"]).to(DEVICE)
        train_loader, val_loader = create_dataloaders(data, batch_size)
        reg_scale = float(np.mean(np.abs(data["y_reg_train"])) + 1e-8)
        _RESOURCES.update(
            class_weights=class_weights,
            train_loader=train_loader,
            val_loader=val_loader,
            reg_scale=reg_scale,
        )
    return _RESOURCES


def objective(trial: Trial, data: Dict[str, np.ndarray], n_epochs: int = 50) -> Tuple[float, float]:
    set_seed(RANDOM_SEED)

    num_classes = int(np.max(data["y_cls_train"])) + 1
    batch_size  = 32

    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])
    if optimizer_name == "SGD":
        learning_rate = trial.suggest_float("learning_rate_sgd", 0.001, 0.02, log=True)
    else:
        learning_rate = trial.suggest_float("learning_rate_adam", 1e-5, 1e-3, log=True)

    hidden_dim            = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_hidden_layers     = 2
    dropout_rate          = trial.suggest_float("dropout_rate", 0.0, 0.2)
    weight_decay          = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
    momentum              = trial.suggest_float("momentum", 0.8, 0.95)
    classification_weight = trial.suggest_float("classification_weight", 0.3, 3.5)
    regression_weight     = trial.suggest_float("regression_weight", 0.3, 2.0)

    resources = get_resources(data, batch_size)

    model = MTLPickleballNet(
        input_dim=data["X_train"].shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        regression_output_dim=data["y_reg_train"].shape[1],
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_batch_norm=False,
    ).to(DEVICE)

    loss_fn = MTLLoss(
        regression_weight=regression_weight,
        classification_weight=classification_weight,
        class_weights=resources["class_weights"],
        use_focal_loss=False,
    )

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    metrics = {}
    for _ in range(n_epochs):
        train_epoch(model, resources["train_loader"], optimizer, loss_fn)
        metrics = evaluate(model, resources["val_loader"], loss_fn)
        norm_mae = metrics["mae"] / resources["reg_scale"]
        neg_f1   = -metrics["f1_score"]
        scheduler.step(neg_f1 + norm_mae)

    del model, optimizer
    gc.collect()

    return metrics["mae"] / resources["reg_scale"], -metrics["f1_score"]


def save_best_params(study: optuna.Study, output_dir: str):
    """Save Pareto-front trials and pick the best balanced trial."""
    best_trials = study.best_trials
    if not best_trials:
        print("No completed trials to save.")
        return

    pareto = []
    for t in best_trials:
        norm_mae, neg_f1 = t.values
        pareto.append({"trial": t.number, "norm_mae": norm_mae, "f1": -neg_f1, "params": t.params})

    # Best balanced = minimise norm_mae - f1 (both equally weighted)
    best = min(pareto, key=lambda d: d["norm_mae"] - d["f1"])

    out = {"pareto_front": pareto, "best_balanced": best}
    path = os.path.join(output_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved best params to: {path}")
    print(f"Best balanced trial #{best['trial']}: NormMAE={best['norm_mae']:.4f}  F1={best['f1']:.4f}")
    print(f"  Params: {best['params']}")


def run_optimization(data_path: str, n_trials: int, output_dir: str):
    print("=" * 60)
    print(f"Optuna Hyperparameter Optimization | Device: {DEVICE}")
    print("=" * 60)
    print(f"Target Trials: {n_trials}")
    print("Mode: Multi-Objective (Normalized MAE, Negative F1)\n")

    data = load_data(data_path)
    print(f"Loaded data — Train: {len(data['X_train']):,}  Val: {len(data['X_val']):,}")

    os.makedirs(output_dir, exist_ok=True)
    db_path    = os.path.join(output_dir, "optuna_study.db")
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="pickleball_mtl_v5_2layer",
        storage=storage_url,
        load_if_exists=True,
        directions=["minimize", "minimize"],
        sampler=TPESampler(seed=RANDOM_SEED, multivariate=True),
    )
    print(f"Study loaded. Completed trials: {len(study.trials)}")

    try:
        study.optimize(
            lambda trial: objective(trial, data),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving results...")

    print("\n" + "=" * 60)
    print("Optimization Finished")
    print("=" * 60)

    print("\nPareto Front:")
    for t in study.best_trials:
        nm, nf1 = t.values
        print(f"  Trial {t.number:4d}: NormMAE={nm:.4f}  F1={-nf1:.4f}  {t.params}")

    save_best_params(study, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Optuna tuner for Pickleball MTL model.")
    parser.add_argument("--data_path",  default=os.path.join(ARTIFACTS_DIR, "splits.npz"))
    parser.add_argument("--n_trials",   type=int, default=5000)
    parser.add_argument("--output_dir", default=ARTIFACTS_DIR)
    args = parser.parse_args()
    run_optimization(args.data_path, args.n_trials, args.output_dir)


if __name__ == "__main__":
    main()
