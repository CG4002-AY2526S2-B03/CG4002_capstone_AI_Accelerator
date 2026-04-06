"""
Export trained PyTorch weights to HLS C header with INT8 quantization.

Per-tensor symmetric quantization:
    scale = max(|W|) / 127
    W_q   = round(W / scale).clip(-128, 127)

Usage:
    python export_weights_int8.py
"""

import argparse
import json
import os
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from model import MTLPickleballNet

DEFAULT_CONFIG = {
    'input_dim': 6,
    'hidden_dim': 512,
    'num_hidden_layers': 2,
    'regression_output_dim': 6,
    'num_classes': 6,
    'dropout_rate': 0.0006,
    'use_batch_norm': False,
}


def quantize_symmetric(W: np.ndarray):
    """Per-tensor symmetric quantization to int8. Returns (W_q, scale)."""
    amax = np.max(np.abs(W))
    if amax == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = float(amax / 127.0)
    W_q = np.round(W / scale).clip(-128, 127).astype(np.int8)
    return W_q, scale


def fmt_int8_array(arr: np.ndarray, name: str, cols: int = 16) -> str:
    flat = arr.flatten().astype(np.int8)
    lines = [f"static const signed char {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), cols):
        chunk = flat[i:i+cols]
        nums = ", ".join(f"{int(v):4d}" for v in chunk)
        comma = "," if i + cols < len(flat) else ""
        lines.append(f"    {nums}{comma}")
    lines.append("};")
    return "\n".join(lines)


def fmt_float_array(arr: np.ndarray, name: str, cols: int = 8) -> str:
    flat = arr.flatten().astype(np.float32)
    lines = [f"static const float {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), cols):
        chunk = flat[i:i+cols]
        nums = ", ".join(f"{v:.8f}f" for v in chunk)
        comma = "," if i + cols < len(flat) else ""
        lines.append(f"    {nums}{comma}")
    lines.append("};")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(HERE, 'artifacts', 'final_model', 'best_model.pth'))
    parser.add_argument("--scaler", type=str, default=os.path.join(HERE, 'artifacts', 'scaler_params.json'))
    parser.add_argument("--out", type=str, default=os.path.join(HERE, '..', '..', 'CG4002_capstone_hls', 'weights.h'))
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    model = MTLPickleballNet(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_hidden_layers=cfg['num_hidden_layers'],
        regression_output_dim=cfg['regression_output_dim'],
        num_classes=cfg['num_classes'],
        dropout_rate=cfg['dropout_rate'],
        use_batch_norm=cfg['use_batch_norm'],
    )
    sd = torch.load(args.model, map_location='cpu', weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    with open(args.scaler) as f:
        scaler = json.load(f)

    IN_DIM = cfg['input_dim']
    HIDDEN = cfg['hidden_dim']
    N_LAYERS = cfg['num_hidden_layers']
    OUT_REG = cfg['regression_output_dim']
    OUT_CLS = cfg['num_classes']
    HEAD_HIDDEN = HIDDEN // 2

    # Fuse BatchNorm into Linear weights (no-op if use_batch_norm=False)
    fused_weights = []
    fused_biases = []

    trunk = model.shared_trunk
    layer_idx = 0
    linear_count = 0
    while layer_idx < len(trunk):
        module = trunk[layer_idx]
        if isinstance(module, torch.nn.Linear):
            W = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            if layer_idx + 1 < len(trunk) and isinstance(trunk[layer_idx + 1], torch.nn.BatchNorm1d):
                bn = trunk[layer_idx + 1]
                gamma = bn.weight.detach().numpy()
                beta = bn.bias.detach().numpy()
                running_mean = bn.running_mean.detach().numpy()
                running_var = bn.running_var.detach().numpy()
                eps = bn.eps
                scale = gamma / np.sqrt(running_var + eps)
                W = W * scale[:, None]
                b = scale * (b - running_mean) + beta
            fused_weights.append(W)
            fused_biases.append(b)
            linear_count += 1
        layer_idx += 1
    assert linear_count == N_LAYERS

    reg_head = model.regression_head
    reg_w1 = reg_head[0].weight.detach().numpy()
    reg_b1 = reg_head[0].bias.detach().numpy()
    reg_w2 = reg_head[2].weight.detach().numpy()
    reg_b2 = reg_head[2].bias.detach().numpy()

    cls_head = model.classification_head
    cls_w1 = cls_head[0].weight.detach().numpy()
    cls_b1 = cls_head[0].bias.detach().numpy()
    cls_w2 = cls_head[2].weight.detach().numpy()
    cls_b2 = cls_head[2].bias.detach().numpy()

    all_weights = {
        'trunk_0': fused_weights[0],
        'trunk_1': fused_weights[1],
        'reg_head_0': reg_w1,
        'reg_head_1': reg_w2,
        'cls_head_0': cls_w1,
        'cls_head_1': cls_w2,
    }
    all_biases = {
        'trunk_0': fused_biases[0],
        'trunk_1': fused_biases[1],
        'reg_head_0': reg_b1,
        'reg_head_1': reg_b2,
        'cls_head_0': cls_b1,
        'cls_head_1': cls_b2,
    }

    quantized = {}
    scales = {}
    total_params = 0
    total_int8 = 0

    print("INT8 Quantization Summary:")
    print(f"{'Layer':<16} {'Shape':<16} {'max|W|':>10} {'scale':>12} {'int8 bytes':>12}")
    print("-" * 70)

    for name, W in all_weights.items():
        W_q, sc = quantize_symmetric(W)
        quantized[name] = W_q
        scales[name] = sc

        W_deq = W_q.astype(np.float32) * sc
        n_params = W.size
        total_params += n_params
        total_int8 += n_params

        print(f"  {name:<14} {str(W.shape):<16} {np.max(np.abs(W)):10.6f} {sc:12.8f} {n_params:12,}")

    print(f"\n  Total int8 weight bytes: {total_int8:,} ({total_int8/1024:.1f} KB)")
    print(f"  Total float32 equivalent: {total_params * 4:,} bytes ({total_params * 4 / 1024:.1f} KB)")
    print(f"  Compression ratio: {total_params * 4 / total_int8:.1f}x")

    bram36k_per = {}
    total_bram = 0
    for name, W_q in quantized.items():
        n = W_q.size
        bram = int(np.ceil(n / 4096))
        bram36k_per[name] = bram
        total_bram += bram
    total_bram += 3

    print(f"\n  Estimated BRAM36K usage: {total_bram} / 216 ({100*total_bram/216:.1f}%)")
    for name, b in bram36k_per.items():
        print(f"    {name}: {b} BRAM36K")
    print(f"    activation buffers: 3 BRAM36K")

    x_mean = np.array(scaler['input_scaler']['mean'], dtype=np.float32)
    x_scale = np.array(scaler['input_scaler']['scale'], dtype=np.float32)
    y_mean = np.array(scaler['regression_scaler']['mean'], dtype=np.float32)
    y_scale = np.array(scaler['regression_scaler']['scale'], dtype=np.float32)

    lines = []
    lines.append("#ifndef WEIGHTS_H")
    lines.append("#define WEIGHTS_H")
    lines.append("")
    lines.append("// Auto-generated INT8-quantized weights (BN fused)")
    lines.append("// Regenerate with: python export_weights_int8.py")
    lines.append(f"// {N_LAYERS} hidden layers, {HIDDEN} units, ReLU6")
    lines.append(f"// {total_params:,} int8 weight params, 4x compression vs float32")
    lines.append(f"// IN={IN_DIM}, HIDDEN={HIDDEN}, HEAD={HEAD_HIDDEN}, REG={OUT_REG}, CLS={OUT_CLS}")
    lines.append("")

    lines.append("// Input scaler: x_norm = (x - mean) / scale")
    lines.append(fmt_float_array(x_mean, "x_scaler_mean"))
    lines.append(fmt_float_array(x_scale, "x_scaler_scale"))
    lines.append("")
    lines.append("// Output scaler (inverse): y_real = y_norm * scale + mean")
    lines.append(fmt_float_array(y_mean, "y_scaler_mean"))
    lines.append(fmt_float_array(y_scale, "y_scaler_scale"))
    lines.append("")

    lines.append("// Per-tensor quantization scales: W_float = W_int8 * qscale")
    for name, sc in scales.items():
        lines.append(f"static const float {name}_qscale = {sc:.10f}f;")
    lines.append("")

    layer_names = ['trunk_0', 'trunk_1']
    prev_dim = IN_DIM
    for i, ln in enumerate(layer_names):
        dim_out = HIDDEN
        lines.append(f"// {ln}: [{dim_out} x {prev_dim}] int8")
        lines.append(fmt_int8_array(quantized[ln], f"{ln}_weight_q"))
        lines.append(fmt_float_array(all_biases[ln], f"{ln}_bias"))
        lines.append("")
        prev_dim = HIDDEN

    lines.append(f"// reg_head_0: [{HEAD_HIDDEN} x {HIDDEN}] int8")
    lines.append(fmt_int8_array(quantized['reg_head_0'], "reg_head_0_weight_q"))
    lines.append(fmt_float_array(all_biases['reg_head_0'], "reg_head_0_bias"))
    lines.append("")
    lines.append(f"// reg_head_1: [{OUT_REG} x {HEAD_HIDDEN}] int8")
    lines.append(fmt_int8_array(quantized['reg_head_1'], "reg_head_1_weight_q"))
    lines.append(fmt_float_array(all_biases['reg_head_1'], "reg_head_1_bias"))
    lines.append("")

    lines.append(f"// cls_head_0: [{HEAD_HIDDEN} x {HIDDEN}] int8")
    lines.append(fmt_int8_array(quantized['cls_head_0'], "cls_head_0_weight_q"))
    lines.append(fmt_float_array(all_biases['cls_head_0'], "cls_head_0_bias"))
    lines.append("")
    lines.append(f"// cls_head_1: [{OUT_CLS} x {HEAD_HIDDEN}] int8")
    lines.append(fmt_int8_array(quantized['cls_head_1'], "cls_head_1_weight_q"))
    lines.append(fmt_float_array(all_biases['cls_head_1'], "cls_head_1_bias"))
    lines.append("")

    lines.append("#endif // WEIGHTS_H")

    with open(args.out, 'w') as f:
        f.write("\n".join(lines))

    print(f"\nWritten to {args.out}")
    print(f"  Weight matrices: int8 (signed char)")
    print(f"  Biases/scalers:  float32")
    print(f"  Quantization:    per-tensor symmetric")


if __name__ == "__main__":
    main()
