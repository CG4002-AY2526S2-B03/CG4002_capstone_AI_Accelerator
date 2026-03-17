import os
import paho.mqtt.client as mqtt
import signal
import numpy as np
import time, json, threading
from ai_event_generator import AIEventGenerator, random_player_ball

BROKER_ADDRESS = "127.0.0.1"
BROKER_PORT = 8883

U96_SUBSCRIBE_TOPIC = "/playerBall"
U96_PUBLISH_TOPIC = "/opponentBall"
CLIENT_ID = "u96-client"

U96_STATUS_TOPIC = "status/u96"
U96_SIGNAL_TOPIC = "system/signal"

_HERE = os.path.dirname(__file__)
CA_CERT = os.path.join(_HERE, "certs", "ca.crt")
CLIENT_CERT = os.path.join(_HERE, "certs", "u96-client.crt")
CLIENT_KEY = os.path.join(_HERE, "certs", "u96-client.key")

is_game_active = True # TODO: set to true as system_coordinator disabled

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"U96 Connected with result code {rc}")
    client.subscribe([(U96_SUBSCRIBE_TOPIC, 0), (U96_SIGNAL_TOPIC, 0)])
    if not is_game_active:
        client.publish(U96_STATUS_TOPIC, "READY", retain=True)

def on_message(client, userdata, msg):
    global is_game_active
    payload = msg.payload.decode('utf-8')

    if msg.topic == U96_SIGNAL_TOPIC:
        if payload == "START":
            is_game_active = True
        return
    elif msg.topic == U96_SUBSCRIBE_TOPIC and is_game_active:
        try:
            print(f"[U96] Received playerBall: {payload}")
            ai_opponent_payload = gen.process_player_ball(payload)
            client.publish(U96_PUBLISH_TOPIC, ai_opponent_payload)
        except Exception as e:
            print(f"Error processing payload: {e}")

def start_random_publisher(client, interval=3.0):
    def _publish_loop():
        while True:
            payload = json.dumps(random_player_ball())
            client.publish(U96_SUBSCRIBE_TOPIC, payload)
            time.sleep(interval)

    t = threading.Thread(target=_publish_loop, daemon=True)
    t.start()

def publish_test_opponent_balls(client, count=10, interval=7.0):
    def _loop():
        time.sleep(1)
        for i in range(count):
            payload = {
                "position": {
                    "x": round(float(np.random.uniform(5, 15)), 4),
                    "y": round(float(np.random.uniform(5, 15)), 4),
                    "z": round(float(np.random.uniform(0.1, 2.0)), 4),
                },
                "velocity": {
                    "vx": round(float(np.random.uniform(-10, 10)), 4),
                    "vy": round(float(np.random.uniform(-10, 10)), 4),
                    "vz": round(float(np.random.uniform(0, 10)), 4),
                },
                "returnSwingType": int(np.random.randint(0, 6))
            }
            client.publish(U96_PUBLISH_TOPIC, json.dumps(payload), qos=1)
            print(f"[U96] Published opponentBall {i+1}/{count}")
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()

gen = AIEventGenerator(
    bitstream = os.path.join(os.path.dirname(__file__), "design_1.bit"),
    use_fpga=True
)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.will_set("/will", payload="U96 DISCONNECTED", qos=1, retain=False)
client.on_connect = on_connect
client.on_message = on_message

client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=CLIENT_KEY
)

client.connect(BROKER_ADDRESS, BROKER_PORT)
print("[U96] Connected MQTT")
client.loop_start()
start_random_publisher(client, interval=5.0)

try:
    signal.pause()
except KeyboardInterrupt:
    print("Shutdown U96.")
    client.loop_stop()
    client.disconnect()
