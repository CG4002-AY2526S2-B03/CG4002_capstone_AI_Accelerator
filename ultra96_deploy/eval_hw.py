#!/usr/bin/env python3

import argparse
import json
import time
import sys
import numpy as np

try:
    from pynq import Overlay
except ImportError:
    print("PYNQ not available. Run this on the Ultra96.")
    sys.exit(1)

from ps_dma_driver import PickleballPredictor, CLASS_NAMES
from power_management import read_power_watts


def main():
    parser = argparse.ArgumentParser(description="Evaluate FPGA hardware accelerator")
    parser.add_argument("bitstream", nargs='?', default="design_1.bit",
                        help="Path to overlay .bit file")
    parser.add_argument("--data", default="splits.npz")
    parser.add_argument("--scaler", default="scaler_params.json")
    parser.add_argument("--n_samples", type=int, default=100, help="Test samples to evaluate")
    args = parser.parse_args()

    # load test data
    data = np.load(args.data)
    X_test_scaled = data['X_test']
    y_cls_test = data['y_cls_test']
    y_reg_test = data['y_reg_test']

    with open(args.scaler) as f:
        scaler = json.load(f)

    x_mean = np.array(scaler['input_scaler']['mean'], dtype=np.float32)
    x_scale = np.array(scaler['input_scaler']['scale'], dtype=np.float32)
    y_mean = np.array(scaler['regression_scaler']['mean'], dtype=np.float32)
    y_scale = np.array(scaler['regression_scaler']['scale'], dtype=np.float32)

    # convert to raw (un-scaled) for FPGA
    X_raw = X_test_scaled * x_scale + x_mean
    y_reg_real = y_reg_test * y_scale + y_mean

    if args.n_samples and args.n_samples < len(X_raw):
        X_raw = X_raw[:args.n_samples]
        y_cls_test = y_cls_test[:args.n_samples]
        y_reg_real = y_reg_real[:args.n_samples]

    N = len(X_raw)
    print(f"FPGA Hardware Evaluation  —  {N} test samples")
    print(f"{'='*55}")

    # load FPGA
    pred = PickleballPredictor(args.bitstream)

    # warmup
    for _ in range(10):
        pred.predict(X_raw[0].tolist())

    # run inference
    hw_times = []
    hw_inf_times = []
    hw_comm_times = []
    hw_reg_all = []
    hw_cls_all = []
    power_samples = []
    for i in range(N):
        reg, cls_idx, cls_name, t_ms, inf_ms, comm_ms = pred.predict_timed(X_raw[i].tolist())
        hw_times.append(t_ms)
        hw_inf_times.append(inf_ms)
        hw_comm_times.append(comm_ms)
        hw_reg_all.append(reg)
        hw_cls_all.append(cls_idx)
        w = read_power_watts()
        if w >= 0:
            power_samples.append(w)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N} done...")

    hw_reg_all = np.array(hw_reg_all)
    hw_cls_all = np.array(hw_cls_all)
    hw_times = np.array(hw_times)
    hw_inf_times = np.array(hw_inf_times)
    hw_comm_times = np.array(hw_comm_times)

    hw_acc = float((hw_cls_all == y_cls_test).mean())
    hw_mae = float(np.mean(np.abs(hw_reg_all - y_reg_real)))

    # confusion matrix (numpy only)
    num_classes = len(CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_cls_test)):
        cm[y_cls_test[i], hw_cls_all[i]] += 1

    # per-class accuracy
    print(f"\n--- Per-Class Accuracy ---")
    for c in range(len(CLASS_NAMES)):
        mask = y_cls_test == c
        if mask.sum() == 0:
            continue
        acc_c = (hw_cls_all[mask] == c).mean()
        print(f"  {CLASS_NAMES[c]:<12} {acc_c:.4f}  ({int(mask.sum())} samples)")

    # Text-based confusion matrix
    print(f"\n--- Confusion Matrix ---")
    print(f"  (rows=true, cols=predicted)")
    print()
    header = "          " + "".join(f"{name[:10]:>10}" for name in CLASS_NAMES)
    print(header)
    for i, true_name in enumerate(CLASS_NAMES):
        row_str = f"  {true_name:<8}"
        for j in range(num_classes):
            row_str += f"{cm[i, j]:>10}"
        print(row_str)
    print()

    # summary
    print(f"\n{'='*55}")
    print(f"  FPGA HARDWARE RESULTS")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-'*42}")
    print(f"  {'Classification Accuracy':<30} {hw_acc:>12.4f}")
    print(f"  {'Regression MAE':<30} {hw_mae:>12.4f}")
    print(f"  {'Mean Total Latency (ms)':<30} {np.mean(hw_times):>12.2f}")
    print(f"  {'Min Total Latency (ms)':<30} {np.min(hw_times):>12.2f}")
    print(f"  {'Max Total Latency (ms)':<30} {np.max(hw_times):>12.2f}")
    print(f"  {'Mean Inference (ms)':<30} {np.mean(hw_inf_times):>12.2f}")
    print(f"  {'Min Inference (ms)':<30} {np.min(hw_inf_times):>12.2f}")
    print(f"  {'Max Inference (ms)':<30} {np.max(hw_inf_times):>12.2f}")
    print(f"  {'Mean Comm Overhead (ms)':<30} {np.mean(hw_comm_times):>12.2f}")
    print(f"  {'Min Comm Overhead (ms)':<30} {np.min(hw_comm_times):>12.2f}")
    print(f"  {'Max Comm Overhead (ms)':<30} {np.max(hw_comm_times):>12.2f}")
    print(f"  {'Throughput (inf/s)':<30} {1e3/np.mean(hw_times):>12.0f}")
    if power_samples:
        pw = np.array(power_samples)
        print(f"  {'Mean Power (W)':<30} {np.mean(pw):>12.2f}")
        print(f"  {'Min Power (W)':<30} {np.min(pw):>12.2f}")
        print(f"  {'Max Power (W)':<30} {np.max(pw):>12.2f}")
    print(f"{'='*55}")

    pred.close()

if __name__ == "__main__":
    main()
