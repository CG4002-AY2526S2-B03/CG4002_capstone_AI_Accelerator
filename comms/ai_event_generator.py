import json
import numpy as np

CLASS_NAMES = ['Drive', 'Drop', 'Dink', 'Lob', 'SpeedUp', 'HandBattle']

INPUT_MEAN  = np.array([10.4265, 10.9689, 0.7985, -0.5195, -2.9106, 3.3363], dtype=np.float32)
INPUT_SCALE = np.array([ 5.7722,  5.8966, 0.3918,  8.1831,  7.7814, 3.1099], dtype=np.float32)


def parse_player_ball(payload: dict) -> list:
    pos = payload['position']
    vel = payload['velocity']
    return [
        float(pos['x']), float(pos['y']), float(pos['z']),
        float(vel['vx']), float(vel['vy']), float(vel['vz']),
    ]

def build_opponent_ball(reg_output: np.ndarray, cls_idx: int) -> dict:
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

class AIEventGenerator:
    def __init__(self, bitstream: str = "design_1.bit", use_fpga: bool = True):
        self.use_fpga = use_fpga
        self.predictor = None

        from ai_ps_dma_driver import PickleballPredictor
        self.predictor = PickleballPredictor(bitstream)


    def process_player_ball(self, payload) -> str:
        if isinstance(payload, str):
            payload = json.loads(payload)

        raw_input = parse_player_ball(payload)

        if self.use_fpga:
            reg, cls_idx, _, latency = self.predictor.predict_timed(raw_input)
        else:
            reg, cls_idx, latency = self._fake_predict(raw_input)

        opponent = build_opponent_ball(reg, cls_idx)
        return json.dumps(opponent)

    def process_player_ball_dict(self, payload) -> dict:
        if isinstance(payload, str):
            payload = json.loads(payload)

        raw_input = parse_player_ball(payload)

        if self.use_fpga:
            reg, cls_idx, _, latency = self.predictor.predict_timed(raw_input)
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
