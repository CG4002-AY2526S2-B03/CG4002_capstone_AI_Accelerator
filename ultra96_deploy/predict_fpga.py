#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
from ps_dma_driver import PickleballPredictor, CLASS_NAMES

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_NAMES = ['ball_x', 'ball_y', 'ball_z', 'ball_vx', 'ball_vy', 'ball_vz']
REG_NAMES   = ['racket_x', 'racket_y', 'racket_z', 'racket_vx', 'racket_vy', 'racket_vz']


def load_test_vectors_by_class():
    # un-scales test set back to raw, groups by class label
    splits_path = os.path.join(SCRIPT_DIR, "splits.npz")
    scaler_path = os.path.join(SCRIPT_DIR, "scaler_params.json")
    data = np.load(splits_path)
    X_test_scaled = data['X_test']
    y_cls_test = data['y_cls_test']

    with open(scaler_path) as f:
        scaler = json.load(f)
    x_mean = np.array(scaler['input_scaler']['mean'], dtype=np.float32)
    x_scale = np.array(scaler['input_scaler']['scale'], dtype=np.float32)
    X_raw = X_test_scaled * x_scale + x_mean

    by_class = {}
    for c in range(len(CLASS_NAMES)):
        idxs = np.where(y_cls_test == c)[0]
        by_class[CLASS_NAMES[c]] = [X_raw[i].tolist() for i in idxs]
    return by_class


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def display_result(raw_input, reg_real, cls_idx, cls_logits, time_ms):
    # pretty-prints input, class probabilities bar chart, regression, and latency
    probs = softmax(cls_logits)

    print(f"\n{'='*50}")
    print(f"  INPUT (raw sensor values)")
    print(f"{'='*50}")
    for name, val in zip(INPUT_NAMES, raw_input):
        print(f"  {name:12s} = {val:10.4f}")

    print(f"\n{'='*50}")
    print(f"  PREDICTION: {CLASS_NAMES[cls_idx]}   ({time_ms:.2f} ms)")
    print(f"{'='*50}")

    print(f"\n  Class probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = '#' * int(probs[i] * 30)
        marker = ' <--' if i == cls_idx else ''
        print(f"    {name:12s} {probs[i]:6.1%}  {bar}{marker}")

    print(f"\n  Regression outputs (predicted racket state):")
    for name, val in zip(REG_NAMES, reg_real):
        print(f"    {name:12s} = {val:10.4f}")
    print()


def main():
    bitstream = sys.argv[1] if len(sys.argv) > 1 else "design_1.bit"
    pred = PickleballPredictor(bitstream)

    print("\n" + "="*50)
    print("  Pickleball Shot Predictor (Ultra96 / FPGA)")
    print("="*50)
    print(f"  Bitstream: {bitstream}")
    print(f"  Input: 6 values [{', '.join(INPUT_NAMES)}]")
    print(f"  Type 'q' to quit, 't' for test batch (5 per class x 6 = 30)\n")

    by_class = load_test_vectors_by_class()
    t_offset = 0

    while True:
        try:
            line = input("Enter 6 values (space-separated): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue
        if line.lower() == 'q':
            print("Bye!")
            break
        if line.lower() == 't':
            test_vectors = []
            for c in CLASS_NAMES:
                samples = by_class[c]
                start = t_offset % len(samples)
                for k in range(5):
                    test_vectors.append((samples[(start + k) % len(samples)], c))
            batch_num = (t_offset // 5) + 1
            print(f"\n  Running test batch #{batch_num} (samples {t_offset}-{t_offset+4} per class)...\n")
            correct = 0
            total = 0
            per_class_correct = {c: 0 for c in CLASS_NAMES}
            per_class_total = {c: 0 for c in CLASS_NAMES}
            latencies = []
            for raw, label in test_vectors:
                reg, cls_idx, cls_name, time_ms, _, _ = pred.predict_timed(raw)
                pred_name = CLASS_NAMES[cls_idx]
                match = pred_name == label
                correct += int(match)
                total += 1
                per_class_total[label] += 1
                per_class_correct[label] += int(match)
                latencies.append(time_ms)
                mark = 'OK' if match else 'WRONG'
                reg_str = ', '.join(f'{v:.2f}' for v in reg)
                print(f"  [{total:2d}/30]  Truth: {label:<12s}  Pred: {pred_name:<12s}  [{mark}]  reg=[{reg_str}]  {time_ms:.2f} ms")
            print(f"\n{'='*55}")
            print(f"  TEST BATCH #{batch_num} RESULTS  ({correct}/{total} correct)")
            print(f"{'='*55}")
            print(f"  {'Class':<14} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
            print(f"  {'-'*40}")
            for c in CLASS_NAMES:
                if per_class_total[c] > 0:
                    acc = per_class_correct[c] / per_class_total[c]
                    print(f"  {c:<14} {per_class_correct[c]:>8} {per_class_total[c]:>6} {acc:>10.0%}")
            print(f"  {'-'*40}")
            print(f"  {'OVERALL':<14} {correct:>8} {total:>6} {correct/total:>10.1%}")
            avg_ms = sum(latencies) / len(latencies)
            print(f"\n  Mean latency: {avg_ms:.2f} ms  ({1e3/avg_ms:.0f} inf/s)")
            print(f"{'='*55}\n")
            t_offset += 5
            continue
        else:
            try:
                raw = [float(v) for v in line.replace(',', ' ').split()]
            except ValueError:
                print("  Error: enter 6 numbers separated by spaces")
                continue

        if len(raw) != 6:
            print(f"  Error: expected 6 values, got {len(raw)}")
            continue

        reg, cls_idx, cls_name, time_ms, _, _ = pred.predict_timed(raw)

        cls_logits = np.array(pred.output_buffer[6:12], dtype=np.float32)

        display_result(raw, reg, cls_idx, cls_logits, time_ms)

    pred.close()


if __name__ == "__main__":
    main()
