#!/usr/bin/env python3

import argparse
import json
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore', message='.*encountered in matmul.*')

try:
    from power_management import read_power_watts
except ImportError:
    def read_power_watts():
        return -1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ['Drive', 'Drop', 'Dink', 'Lob', 'SpeedUp', 'HandBattle']


def confusion_matrix_numpy(y_true, y_pred, num_classes):
    # Compute confusion matrix using numpy only (no sklearn)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def numpy_inference(X_raw: np.ndarray, weights: dict) -> tuple:
    # forward pass in float64 (no BN, 2 trunk layers)
    def relu6(x):
        return np.clip(x, 0, 6)

    x = (X_raw.astype(np.float64) - weights['x_mean']) / weights['x_scale']

    h = x
    for i in range(2):
        W = weights[f'trunk_{i}_weight']
        b = weights[f'trunk_{i}_bias']
        h = relu6(h @ W.T + b)

    h_reg = relu6(h @ weights['reg_head_0_weight'].T + weights['reg_head_0_bias'])
    reg_norm = h_reg @ weights['reg_head_1_weight'].T + weights['reg_head_1_bias']
    reg_real = reg_norm * weights['y_scale'] + weights['y_mean']

    h_cls = relu6(h @ weights['cls_head_0_weight'].T + weights['cls_head_0_bias'])
    cls_logits = h_cls @ weights['cls_head_1_weight'].T + weights['cls_head_1_bias']
    cls_pred = np.argmax(cls_logits, axis=-1)

    return reg_real, cls_pred, cls_logits


def main():
    parser = argparse.ArgumentParser(description="Evaluate software (NumPy) inference on Ultra96 ARM CPU")
    parser.add_argument("--data", default=os.path.join(SCRIPT_DIR, "splits.npz"))
    parser.add_argument("--weights", default=os.path.join(SCRIPT_DIR, "fused_weights.npz"))
    parser.add_argument("--scaler", default=os.path.join(SCRIPT_DIR, "scaler_params.json"))
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    # load test data
    data = np.load(args.data)
    X_test_scaled = data['X_test']
    y_cls_test = data['y_cls_test']
    y_reg_test = data['y_reg_test']

    with open(args.scaler) as f:
        scaler = json.load(f)

    x_mean = np.array(scaler['input_scaler']['mean'], dtype=np.float64)
    x_scale = np.array(scaler['input_scaler']['scale'], dtype=np.float64)
    y_mean = np.array(scaler['regression_scaler']['mean'], dtype=np.float64)
    y_scale = np.array(scaler['regression_scaler']['scale'], dtype=np.float64)

    X_raw = X_test_scaled * x_scale + x_mean
    y_reg_real = y_reg_test * y_scale + y_mean

    if args.n_samples and args.n_samples < len(X_raw):
        X_raw = X_raw[:args.n_samples]
        y_cls_test = y_cls_test[:args.n_samples]
        y_reg_real = y_reg_real[:args.n_samples]

    N = len(X_raw)

    # load pre-exported fused weights (no PyTorch needed)
    weights = dict(np.load(args.weights))

    print(f"Software (NumPy) Evaluation on Ultra96 CPU  —  {N} samples")
    print(f"{'='*55}")

    sw_times = []
    sw_reg_all = []
    sw_cls_all = []
    power_samples = []
    for i in range(N):
        t0 = time.perf_counter()
        reg, cls, _ = numpy_inference(X_raw[i:i+1], weights)
        t1 = time.perf_counter()
        sw_times.append((t1 - t0) * 1e3)
        sw_reg_all.append(reg[0])
        sw_cls_all.append(cls[0])
        w = read_power_watts()
        if w >= 0:
            power_samples.append(w)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N} done...")

    sw_reg_all = np.array(sw_reg_all)
    sw_cls_all = np.array(sw_cls_all)
    sw_times = np.array(sw_times)

    sw_acc = float((sw_cls_all == y_cls_test).mean())
    sw_mae = float(np.mean(np.abs(sw_reg_all - y_reg_real)))

    # Confusion matrix (numpy only, no sklearn)
    cm = confusion_matrix_numpy(y_cls_test, sw_cls_all, len(CLASS_NAMES))

    # per-class accuracy
    print(f"\n--- Per-Class Accuracy ---")
    for c in range(len(CLASS_NAMES)):
        mask = y_cls_test == c
        if mask.sum() == 0:
            continue
        acc_c = (sw_cls_all[mask] == c).mean()
        print(f"  {CLASS_NAMES[c]:<12} {acc_c:.4f}  ({int(mask.sum())} samples)")

    # Print confusion matrix in text form
    print(f"\n--- Confusion Matrix ---")
    print(f"  (rows=true label, cols=predicted label)")
    print()
    
    # Header with class names
    header = "         " + "".join(f"{name[:7]:>10}" for name in CLASS_NAMES)
    print(header)
    
    # Matrix rows
    for i, true_name in enumerate(CLASS_NAMES):
        row_str = f"{true_name:<8}"
        for j in range(len(CLASS_NAMES)):
            row_str += f"{cm[i, j]:>10}"
        print(row_str)
    
    print()

    # summary
    print(f"\n{'='*55}")
    print(f"  ULTRA96 CPU SOFTWARE RESULTS")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*42}")
    print(f"  {'Classification Accuracy':<30} {sw_acc:>12.4f}")
    print(f"  {'Regression MAE':<30} {sw_mae:>12.4f}")
    print(f"  {'Mean Latency (ms)':<30} {np.mean(sw_times):>12.2f}")
    print(f"  {'Min Latency (ms)':<30} {np.min(sw_times):>12.2f}")
    print(f"  {'Max Latency (ms)':<30} {np.max(sw_times):>12.2f}")
    print(f"  {'Throughput (inf/s)':<30} {1e3/np.mean(sw_times):>12.0f}")
    if power_samples:
        pw = np.array(power_samples)
        print(f"  {'Mean Power (W)':<30} {np.mean(pw):>12.2f}")
        print(f"  {'Min Power (W)':<30} {np.min(pw):>12.2f}")
        print(f"  {'Max Power (W)':<30} {np.max(pw):>12.2f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
