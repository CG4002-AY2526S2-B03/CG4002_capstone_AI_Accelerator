#!/usr/bin/env python3


import json
import socket
import time
import threading
import numpy as np

CLASS_NAMES = ['Drive', 'Drop', 'Dink', 'Lob', 'SpeedUp', 'HandBattle']

INPUT_MEAN  = np.array([-0.00898717,  0.86348623,  1.22588217,  0.04948143,  2.95032215, 10.96861172], dtype=np.float32)
INPUT_SCALE = np.array([ 1.97547972,  0.23852809,  1.83424926,  5.59901857,  2.58921480,  4.79342556], dtype=np.float32)


def parse_player_ball(payload: dict) -> list:
    # extracts 6 floats [x,y,z,vx,vy,vz] from /playerBall JSON
    pos = payload['position']
    vel = payload['velocity']
    return [
        float(pos['x']), float(pos['y']), float(pos['z']),
        float(vel['vx']), float(vel['vy']), float(vel['vz']),
    ]


def build_opponent_ball(reg_output: np.ndarray, cls_idx: int) -> dict:
    # packs regression output + class index into /opponentBall JSON
    return {
        'position': {
            'x':  round(float(reg_output[0]), 4),
            'y':  round(float(reg_output[1]), 4),
            'z':  round(float(reg_output[2]), 4),
        },
        'velocity': {
            'vx': round(float(reg_output[3]), 4),
            'vy': round(float(reg_output[4]), 4),
            'vz': round(float(reg_output[5]), 4),
        },
        'returnSwingType': int(cls_idx),
    }


def random_player_ball() -> dict:
    # generates a random plausible /playerBall for testing
    z = np.random.uniform(-2.0, 2.0, size=6).astype(np.float32)
    raw = INPUT_MEAN + z * INPUT_SCALE
    raw[2] = max(raw[2], 0.05)
    return {
        'position': {'x': round(float(raw[0]), 4),
                     'y': round(float(raw[1]), 4),
                     'z': round(float(raw[2]), 4)},
        'velocity': {'vx': round(float(raw[3]), 4),
                     'vy': round(float(raw[4]), 4),
                     'vz': round(float(raw[5]), 4)},
    }


def _softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

class AIEventGenerator:
    # wraps FPGA (or fake) predictor; processes /playerBall -> /opponentBall
    def __init__(self, bitstream: str = "design_1.bit", use_fpga: bool = True):
        self.use_fpga = use_fpga
        self.predictor = None

        if use_fpga:
            from ai_ps_dma_driver import PickleballPredictor
            self.predictor = PickleballPredictor(bitstream)


    def process_player_ball(self, payload) -> str:
        if isinstance(payload, str):
            payload = json.loads(payload)

        raw_input = parse_player_ball(payload)

        if self.use_fpga:
            reg, cls_idx, _, latency = self.predictor.predict_timed(raw_input)  # returns (reg, idx, name, time_us)
        else:
            reg, cls_idx, latency = self._fake_predict(raw_input)

        opponent = build_opponent_ball(reg, cls_idx)
        return json.dumps(opponent)

    def process_player_ball_dict(self, payload) -> dict:
        if isinstance(payload, str):
            payload = json.loads(payload)

        raw_input = parse_player_ball(payload)

        if self.use_fpga:
            reg, cls_idx, _, latency = self.predictor.predict_timed(raw_input)  # returns (reg, idx, name, time_us)
        else:
            reg, cls_idx, latency = self._fake_predict(raw_input)

        return build_opponent_ball(reg, cls_idx)

    def generate_random(self) -> str:
        return self.process_player_ball(random_player_ball())

    def close(self):
        if self.predictor is not None:
            self.predictor.close()

    def _fake_predict(self, raw_input: list):
        cls_idx = np.random.randint(0, len(CLASS_NAMES))
        reg = INPUT_MEAN + np.random.randn(6).astype(np.float32) * INPUT_SCALE * 0.5
        return reg, cls_idx, 0.0


def tcp_serve(gen: AIEventGenerator, host: str = "0.0.0.0", port: int = 3000):
    # accepts TCP connections and dispatches newline-delimited JSON
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"[TCP] Listening on {host}:{port}  (waiting for laptop relay...)")

    while True:
        conn, addr = srv.accept()
        print(f"[TCP] Connected: {addr}")
        threading.Thread(target=_handle_client, args=(gen, conn, addr), daemon=True).start()


def _handle_client(gen: AIEventGenerator, conn: socket.socket, addr):
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    t0 = time.perf_counter()
                    response = gen.process_player_ball(line.decode("utf-8"))
                    dt = (time.perf_counter() - t0) * 1000
                    conn.sendall((response + "\n").encode("utf-8"))
                    print(f"[TCP] {addr} → processed in {dt:.1f} ms")
                except Exception as e:
                    err = json.dumps({"error": str(e)}) + "\n"
                    conn.sendall(err.encode("utf-8"))
                    print(f"[TCP] {addr} → ERROR: {e}")
    except ConnectionResetError:
        pass
    finally:
        conn.close()
        print(f"[TCP] Disconnected: {addr}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ultra96 AI Event Generator")
    parser.add_argument("--fpga", action="store_true", help="Use real FPGA inference")
    parser.add_argument("--port", type=int, default=3000, help="TCP listen port (default 3000)")
    parser.add_argument("--demo", action="store_true", help="Run 5 fake events and exit (no TCP)")
    args = parser.parse_args()

    gen = AIEventGenerator(use_fpga=args.fpga)

    if args.demo:
        print(f"AI Event Generator  |  FPGA: {args.fpga}")
        print("=" * 55)
        for i in range(5):
            pb = random_player_ball()
            ob_json = gen.process_player_ball(pb)
            ob = json.loads(ob_json)
            print(f"\nEvent {i+1}:")
            print(f"  /playerBall  → {json.dumps(pb)}")
            print(f"  /opponentBall← {ob_json}")
            print(f"    swing type: {ob['returnSwingType']} ({CLASS_NAMES[ob['returnSwingType']]})")
        gen.close()
        print("\nDone.")
    else:
        print(f"AI Event Generator  |  FPGA: {args.fpga}  |  Port: {args.port}")
        print("=" * 55)
        print("Ctrl+C to stop.\n")
        try:
            tcp_serve(gen, port=args.port)
        except KeyboardInterrupt:
            print("\n[TCP] Shutting down.")
        finally:
            gen.close()
