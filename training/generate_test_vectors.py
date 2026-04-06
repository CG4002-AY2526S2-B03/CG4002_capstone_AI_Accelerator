"""
Generate HLS test vectors with INT8-quantized golden reference.

Usage:
    python generate_test_vectors.py
    python generate_test_vectors.py --n_per_class 5
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from model import MTLPickleballNet

HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = ['Drive', 'Drop', 'Dink', 'Lob', 'SpeedUp', 'HandBattle']

SEED = 45

DEFAULT_CONFIG = {
    'input_dim': 6,
    'hidden_dim': 512,
    'num_hidden_layers': 2,
    'regression_output_dim': 6,
    'num_classes': 6,
    'dropout_rate': 0.0006,
    'use_batch_norm': False,
}


def quantize_symmetric(w: np.ndarray):
    amax = np.abs(w).max()
    if amax < 1e-12:
        return np.zeros_like(w, dtype=np.int8), 0.0
    scale = amax / 127.0
    w_q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_q, float(scale)


def relu6(x):
    return np.clip(x, 0.0, 6.0)


def quantized_linear(x, w_q, qscale, bias, activation='relu6'):
    w_f = w_q.astype(np.float32)
    y = x @ w_f.T
    y = y * qscale + bias
    if activation == 'relu6':
        y = relu6(y)
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(HERE, 'artifacts', 'final_model', 'best_model.pth'))
    parser.add_argument("--data", default=os.path.join(HERE, 'artifacts', 'splits.npz'))
    parser.add_argument("--scaler", default=os.path.join(HERE, 'artifacts', 'scaler_params.json'))
    parser.add_argument("--n_per_class", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(HERE, '..', '..', 'CG4002_capstone_hls', 'test_vectors.h'))
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Seed: {SEED}\n")

    data = np.load(args.data)
    X_test = data['X_test']
    y_cls_test = data['y_cls_test']
    y_reg_test = data['y_reg_test']

    with open(args.scaler) as f:
        scaler = json.load(f)

    x_mean = np.array(scaler['input_scaler']['mean'], dtype=np.float32)
    x_scale = np.array(scaler['input_scaler']['scale'], dtype=np.float32)
    y_mean = np.array(scaler['regression_scaler']['mean'], dtype=np.float32)
    y_scale = np.array(scaler['regression_scaler']['scale'], dtype=np.float32)

    cfg = DEFAULT_CONFIG
    model = MTLPickleballNet(**{k: cfg[k] for k in [
        'input_dim', 'hidden_dim', 'num_hidden_layers',
        'regression_output_dim', 'num_classes', 'dropout_rate', 'use_batch_norm'
    ]})
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    model.eval()

    # Extract and quantize trunk layers
    trunk = model.shared_trunk
    layers_info = []
    i = 0
    layer_idx = 0
    while i < len(trunk):
        mod = trunk[i]
        if isinstance(mod, nn.Linear):
            W_f = mod.weight.detach().numpy()
            b_f = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(mod.out_features, dtype=np.float32)
            if i + 1 < len(trunk) and isinstance(trunk[i + 1], nn.BatchNorm1d):
                bn = trunk[i + 1]
                inv_std = bn.weight.detach().numpy() / np.sqrt(bn.running_var.detach().numpy() + bn.eps)
                W_f = (W_f * inv_std[:, None]).astype(np.float32)
                b_f = ((b_f - bn.running_mean.detach().numpy()) * inv_std + bn.bias.detach().numpy()).astype(np.float32)
            W_q, qscale = quantize_symmetric(W_f)
            layers_info.append((W_q, qscale, b_f))
            print(f"  trunk_{layer_idx}: shape {W_f.shape}, qscale={qscale:.8f}")
            layer_idx += 1
        i += 1

    def extract_head(head_seq):
        head_layers = []
        for mod in head_seq:
            if isinstance(mod, nn.Linear):
                W_f = mod.weight.detach().numpy()
                b_f = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(mod.out_features, dtype=np.float32)
                W_q, qscale = quantize_symmetric(W_f)
                head_layers.append((W_q, qscale, b_f))
        return head_layers

    reg_layers = extract_head(model.regression_head)
    cls_layers = extract_head(model.classification_head)

    for i, (_, qs, _) in enumerate(reg_layers):
        print(f"  reg_head_{i}: qscale={qs:.8f}")
    for i, (_, qs, _) in enumerate(cls_layers):
        print(f"  cls_head_{i}: qscale={qs:.8f}")
    print()

    # Pick n_per_class samples from test set
    selected_indices = []
    for c in range(len(CLASS_NAMES)):
        idxs = np.where(y_cls_test == c)[0]
        if len(idxs) == 0:
            print(f"  WARNING: No test samples for class {c} ({CLASS_NAMES[c]})")
            continue
        chosen = np.random.choice(idxs, size=min(args.n_per_class, len(idxs)), replace=False)
        selected_indices.extend(chosen.tolist())

    N = len(selected_indices)
    print(f"Selected {N} test samples ({args.n_per_class}/class × {len(CLASS_NAMES)} classes)\n")

    X_sel = X_test[selected_indices]
    y_cls_sel = y_cls_test[selected_indices]

    # Reconstruct raw inputs from scaled data
    X_raw = X_sel * x_scale + x_mean
    X_scaled = (X_raw - x_mean) / x_scale

    # Quantized inference
    h = X_scaled.astype(np.float32)
    for W_q, qscale, bias in layers_info:
        h = quantized_linear(h, W_q, qscale, bias, activation='relu6')

    h_reg = h.copy()
    for li, (W_q, qscale, bias) in enumerate(reg_layers):
        act = 'relu6' if li < len(reg_layers) - 1 else 'none'
        h_reg = quantized_linear(h_reg, W_q, qscale, bias, activation=act)
    reg_real = h_reg * y_scale + y_mean

    h_cls = h.copy()
    for li, (W_q, qscale, bias) in enumerate(cls_layers):
        act = 'relu6' if li < len(cls_layers) - 1 else 'none'
        h_cls = quantized_linear(h_cls, W_q, qscale, bias, activation=act)
    cls_pred = h_cls.argmax(axis=1)

    # Generate C header
    lines = []
    lines.append("#ifndef TEST_VECTORS_H")
    lines.append("#define TEST_VECTORS_H")
    lines.append("")
    lines.append(f"// Auto-generated test vectors ({N} samples, {args.n_per_class}/class)")
    lines.append(f"// Golden reference from INT8-quantized inference, seed={SEED}")
    lines.append("")
    lines.append(f"#define N_TESTS {N}")
    lines.append("")

    lines.append(f"const float test_inputs[{N}][IN_DIM] = {{")
    for i, idx in enumerate(selected_indices):
        vals = ", ".join(f"{v:.6f}f" for v in X_raw[i])
        lines.append(f"    {{{vals}}}, // Sample {i}: class {int(y_cls_sel[i])} ({CLASS_NAMES[int(y_cls_sel[i])]})")
    lines.append("};")
    lines.append("")

    lines.append(f"const int expected_true_cls[{N}] = {{")
    lines.append("    " + ", ".join(str(int(c)) for c in y_cls_sel))
    lines.append("};")
    lines.append("")

    lines.append(f"const int expected_pred_cls[{N}] = {{")
    lines.append("    " + ", ".join(str(int(c)) for c in cls_pred))
    lines.append("};")
    lines.append("")

    lines.append(f"const float expected_reg[{N}][OUT_REG] = {{")
    for i in range(N):
        vals = ", ".join(f"{v:.6f}f" for v in reg_real[i])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    lines.append(f"const float expected_cls_logits[{N}][OUT_CLS] = {{")
    for i in range(N):
        vals = ", ".join(f"{v:.6f}f" for v in h_cls[i])
        lines.append(f"    {{{vals}}},")
    lines.append("};")
    lines.append("")

    lines.append("#endif // TEST_VECTORS_H")

    with open(args.out, 'w') as f:
        f.write("\n".join(lines))

    print(f"Written to {args.out}")

    correct = (cls_pred == y_cls_sel).sum()
    print(f"Quantized model accuracy: {correct}/{N} ({100*correct/N:.1f}%)")

    with torch.no_grad():
        X_t = torch.FloatTensor(X_sel)
        _, cls_logits_fp32 = model(X_t)
        cls_pred_fp32 = cls_logits_fp32.argmax(dim=1).numpy()
    correct_fp32 = (cls_pred_fp32 == y_cls_sel).sum()
    print(f"Float32 PyTorch accuracy:  {correct_fp32}/{N} ({100*correct_fp32/N:.1f}%)")
    disagree = (cls_pred != cls_pred_fp32).sum()
    if disagree > 0:
        print(f"  Quantization changed {disagree} predictions")


if __name__ == "__main__":
    main()
