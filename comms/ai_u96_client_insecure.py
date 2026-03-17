import os
import paho.mqtt.client as mqtt
import signal
from ai_event_generator import AIEventGenerator

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

is_game_active = False

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
            print(">>> START SIGNAL RECEIVED. FPGA Processing Enabled.")
            is_game_active = True
        return
    elif msg.topic == U96_SUBSCRIBE_TOPIC and is_game_active:
        ai_opponent_payload = gen.process_player_ball(payload)
        client.publish(U96_PUBLISH_TOPIC, ai_opponent_payload)

gen = AIEventGenerator(
    bitstream = os.path.join(os.path.dirname(__file__), "design_1.bit"),
    use_fpga=True
)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.will_set("/will", payload="U96 DISCONNECTED", qos=1, retain=False)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_ADDRESS, BROKER_PORT)
client.loop_start()

try:
    signal.pause()
except KeyboardInterrupt:
    print("Shutdown U96.")
    client.loop_stop()
    client.disconnect()
