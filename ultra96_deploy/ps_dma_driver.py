#!/usr/bin/env python3

import time
import numpy as np

try:
    from pynq import Overlay, allocate, MMIO
except ImportError:
    print("WARNING: PYNQ not available. This module only works on Ultra96.")
    Overlay = None
    allocate = None
    MMIO = None


CLASS_NAMES = ['Drive', 'Drop', 'Dink', 'Lob', 'SpeedUp', 'HandBattle']

N_INPUT = 6
N_REG_OUTPUT = 6
N_CLS_OUTPUT = 6
N_TOTAL_OUTPUT = N_REG_OUTPUT + N_CLS_OUTPUT

# DMA Register Offsets (Direct Register / Simple Mode)
# PYNQ's DMA class incorrectly uses _SGDMAChannel even when SG is disabled,
# so we program the registers directly via MMIO.
MM2S_DMACR  = 0x00
MM2S_DMASR  = 0x04
MM2S_SA     = 0x18
MM2S_SA_MSB = 0x1C
MM2S_LENGTH = 0x28
S2MM_DMACR  = 0x30
S2MM_DMASR  = 0x34
S2MM_DA     = 0x48
S2MM_DA_MSB = 0x4C
S2MM_LENGTH = 0x58


class PickleballPredictor:
    # raw-MMIO DMA driver: sends 6 floats to FPGA, reads back 12 (6 reg + 6 cls logits)

    def __init__(
        self,
        bitstream_path: str = "design_1.bit",
        dma_name: str = "axi_dma_0",
    ):
        if Overlay is None:
            raise RuntimeError("PYNQ library not available. Run this on Ultra96.")

        print(f"Loading overlay: {bitstream_path}")
        self.overlay = Overlay(bitstream_path)

        dma_addr = self.overlay.ip_dict[dma_name]['phys_addr']
        hls_addr = self.overlay.ip_dict['pb_predict_0']['phys_addr']
        self.dma_mmio = MMIO(dma_addr, 0x100)
        self.hls_mmio = MMIO(hls_addr, 0x100)

        self.input_buffer  = allocate(shape=(N_INPUT,),        dtype=np.float32)
        self.output_buffer = allocate(shape=(N_TOTAL_OUTPUT,), dtype=np.float32)

        self._reset_dma()

        print(f"Overlay loaded. DMA (raw MMIO) ready.")
        self._print_ip_info()

    def _reset_dma(self):
        self.dma_mmio.write(MM2S_DMACR, 0x04)
        self.dma_mmio.write(S2MM_DMACR, 0x04)
        time.sleep(0.001)

    def _print_ip_info(self):
        print(f"  IPs in overlay: {list(self.overlay.ip_dict.keys())}")

    def predict(self, raw_input: list) -> tuple:
        assert len(raw_input) == N_INPUT, f"Expected {N_INPUT} inputs, got {len(raw_input)}"

        self.dma_mmio.write(MM2S_DMACR, 0x0000)
        self.dma_mmio.write(S2MM_DMACR, 0x0000)
        self.dma_mmio.write(MM2S_DMASR, 0x7000)
        self.dma_mmio.write(S2MM_DMASR, 0x7000)

        for i in range(N_INPUT):
            self.input_buffer[i] = np.float32(raw_input[i])
        self.input_buffer.flush()

        in_phys  = self.input_buffer.physical_address
        out_phys = self.output_buffer.physical_address

        self.dma_mmio.write(S2MM_DMACR, 0x0001)
        self.dma_mmio.write(S2MM_DA,     out_phys & 0xFFFFFFFF)
        self.dma_mmio.write(S2MM_DA_MSB, (out_phys >> 32) & 0xFFFFFFFF)
        self.dma_mmio.write(S2MM_LENGTH, self.output_buffer.nbytes)

        self.hls_mmio.write(0x00, 0x01)

        self.dma_mmio.write(MM2S_DMACR, 0x0001)
        self.dma_mmio.write(MM2S_SA,     in_phys & 0xFFFFFFFF)
        self.dma_mmio.write(MM2S_SA_MSB, (in_phys >> 32) & 0xFFFFFFFF)
        self.dma_mmio.write(MM2S_LENGTH, self.input_buffer.nbytes)

        TIMEOUT_US = 500_000
        t_start = time.perf_counter()
        while True:
            status = self.dma_mmio.read(S2MM_DMASR)
            if status & 0x0002:
                break
            if status & 0x0070:
                self._reset_dma()
                raise RuntimeError(f"DMA error: S2MM_DMASR=0x{status:08x}")
            if (time.perf_counter() - t_start) * 1e6 > TIMEOUT_US:
                mm2s_sr = self.dma_mmio.read(0x04)
                hls_cr  = self.hls_mmio.read(0x00)
                self._reset_dma()
                raise TimeoutError(
                    f"DMA poll timeout ({TIMEOUT_US} µs). "
                    f"S2MM_SR=0x{status:08x}, MM2S_SR=0x{mm2s_sr:08x}, HLS_CR=0x{hls_cr:08x}"
                )

        self.output_buffer.invalidate()
        reg_output = np.array(self.output_buffer[:N_REG_OUTPUT],               dtype=np.float32)
        cls_logits = np.array(self.output_buffer[N_REG_OUTPUT:N_TOTAL_OUTPUT], dtype=np.float32)
        cls_index  = int(np.argmax(cls_logits))
        cls_name   = CLASS_NAMES[cls_index] if cls_index < len(CLASS_NAMES) else f"class_{cls_index}"

        return reg_output, cls_index, cls_name

    def predict_timed(self, raw_input: list) -> tuple:
        # Returns (reg_output, cls_index, cls_name, total_ms, inference_ms, comm_ms)
        assert len(raw_input) == N_INPUT, f"Expected {N_INPUT} inputs, got {len(raw_input)}"

        t_total_start = time.perf_counter()

        self.dma_mmio.write(MM2S_DMACR, 0x0000)
        self.dma_mmio.write(S2MM_DMACR, 0x0000)
        self.dma_mmio.write(MM2S_DMASR, 0x7000)
        self.dma_mmio.write(S2MM_DMASR, 0x7000)

        for i in range(N_INPUT):
            self.input_buffer[i] = np.float32(raw_input[i])
        self.input_buffer.flush()

        in_phys  = self.input_buffer.physical_address
        out_phys = self.output_buffer.physical_address

        self.dma_mmio.write(S2MM_DMACR, 0x0001)
        self.dma_mmio.write(S2MM_DA,     out_phys & 0xFFFFFFFF)
        self.dma_mmio.write(S2MM_DA_MSB, (out_phys >> 32) & 0xFFFFFFFF)
        self.dma_mmio.write(S2MM_LENGTH, self.output_buffer.nbytes)

        self.hls_mmio.write(0x00, 0x01)

        self.dma_mmio.write(MM2S_DMACR, 0x0001)
        self.dma_mmio.write(MM2S_SA,     in_phys & 0xFFFFFFFF)
        self.dma_mmio.write(MM2S_SA_MSB, (in_phys >> 32) & 0xFFFFFFFF)

        t_inference_start = time.perf_counter()
        self.dma_mmio.write(MM2S_LENGTH, self.input_buffer.nbytes)

        TIMEOUT_US = 500_000
        while True:
            status = self.dma_mmio.read(S2MM_DMASR)
            if status & 0x0002:
                break
            if status & 0x0070:
                self._reset_dma()
                raise RuntimeError(f"DMA error: S2MM_DMASR=0x{status:08x}")
            if (time.perf_counter() - t_inference_start) * 1e6 > TIMEOUT_US:
                mm2s_sr = self.dma_mmio.read(0x04)
                hls_cr  = self.hls_mmio.read(0x00)
                self._reset_dma()
                raise TimeoutError(
                    f"DMA poll timeout ({TIMEOUT_US} µs). "
                    f"S2MM_SR=0x{status:08x}, MM2S_SR=0x{mm2s_sr:08x}, HLS_CR=0x{hls_cr:08x}"
                )
        t_inference_end = time.perf_counter()

        self.output_buffer.invalidate()
        reg_output = np.array(self.output_buffer[:N_REG_OUTPUT],               dtype=np.float32)
        cls_logits = np.array(self.output_buffer[N_REG_OUTPUT:N_TOTAL_OUTPUT], dtype=np.float32)
        cls_index  = int(np.argmax(cls_logits))
        cls_name   = CLASS_NAMES[cls_index] if cls_index < len(CLASS_NAMES) else f"class_{cls_index}"

        t_total_end  = time.perf_counter()
        total_ms     = (t_total_end     - t_total_start)     * 1e3
        inference_ms = (t_inference_end - t_inference_start) * 1e3
        comm_ms      = total_ms - inference_ms

        return reg_output, cls_index, cls_name, total_ms, inference_ms, comm_ms

    def predict_batch(self, inputs: np.ndarray) -> list:
        results = []
        for row in inputs:
            results.append(self.predict(row.tolist()))
        return results

    def benchmark(self, n_iterations: int = 1000) -> dict:
        test_input = [16.33, 18.370001, 1.052669, -11.010001, -8.82, 0.632585]
        times = []
        for _ in range(10):
            self.predict(test_input)
        for _ in range(n_iterations):
            _, _, _, t, inf_t, comm_t = self.predict_timed(test_input)
            times.append(t)
        times = np.array(times)
        return {
            'n_iterations': n_iterations,
            'mean_ms': float(np.mean(times)),
            'std_ms':  float(np.std(times)),
            'min_ms':  float(np.min(times)),
            'max_ms':  float(np.max(times)),
            'p50_ms':  float(np.percentile(times, 50)),
            'p99_ms':  float(np.percentile(times, 99)),
            'throughput_hz': float(1e3 / np.mean(times)),
        }

    def close(self):
        del self.input_buffer
        del self.output_buffer


if __name__ == "__main__":
    import sys

    bitstream = sys.argv[1] if len(sys.argv) > 1 else "design_1.bit"
    pred = PickleballPredictor(bitstream)

    test_input = [16.33, 18.370001, 1.052669, -11.010001, -8.82, 0.632585]

    reg, cls_idx, cls_name, time_ms, inf_ms, comm_ms = pred.predict_timed(test_input)
    print(f"\nInput:      {test_input}")
    print(f"Reg output: {reg}")
    print(f"Predicted:  {cls_name} (class {cls_idx})")
    print(f"Total:      {time_ms:.2f} ms")
    print(f"Inference:  {inf_ms:.2f} ms")
    print(f"Comm:       {comm_ms:.2f} ms")

    expected_reg = np.array([6.213955, 9.685316, 0.728952, 4.642068, 4.561520, 1.831140])
    reg_diff = np.abs(reg - expected_reg)
    print(f"\nReg diff vs golden: {reg_diff}")
    print(f"Max reg diff:       {reg_diff.max():.6f}")

    print("\nRunning benchmark (1000 iterations)...")
    stats = pred.benchmark(1000)
    print(f"  Mean:       {stats['mean_ms']:.2f} ms")
    print(f"  Min/Max:    {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
    print(f"  P50/P99:    {stats['p50_ms']:.2f} / {stats['p99_ms']:.2f} ms")
    print(f"  Throughput: {stats['throughput_hz']:.0f} inferences/sec")

    pred.close()
