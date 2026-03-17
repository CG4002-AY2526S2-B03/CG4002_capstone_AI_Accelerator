import os
import paho.mqtt.client as mqtt
import time
import json

BROKER_ADDRESS = "127.0.0.1"
BROKER_PORT = 1883

U96_SUBSCRIBE_TOPIC = "/playerBall"
U96_PUBLISH_TOPIC = "/opponentBall"
CLIENT_ID = "u96-client"

_HERE = os.path.dirname(__file__)
CA_CERT = os.path.join(_HERE, "certs", "ca.crt")
CLIENT_CERT = os.path.join(_HERE, "certs", "u96-client.crt")
CLIENT_KEY = os.path.join(_HERE, "certs", "u96-client.key")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"U96 Connected with result code {rc}")
    client.subscribe(U96_SUBSCRIBE_TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))

        if "vx" in payload:
            x = payload.get("x")
            y = payload.get("y")
            z = payload.get("z")
            vx = payload.get("vx")
            vy = payload.get("vy")
            vz = payload.get("vz")
            print(f"Received message from {U96_SUBSCRIBE_TOPIC}: x={x}, y={y}, z={z}, vx={vx}, vy={vy}, vz={vz}")
        else:
            print(f"[ERROR] Unexpected message format from {U96_SUBSCRIBE_TOPIC}: {payload}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from message: {msg.payload.decode('utf-8')}")
    except Exception as e:
        print(f"Error processing message: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
client.will_set("/will", payload="U96 has disconnected", qos=1, retain=False)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_ADDRESS, BROKER_PORT)
client.loop_start()

try:
    while True:
        ai_data = {
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "vx": 0.5,
            "vy": 0.5,
            "vz": 0.5,
            "returnSwing": 0
        }
        ai_payload = json.dumps(ai_data)
        client.publish(U96_PUBLISH_TOPIC, ai_payload)
        time.sleep(5)
except KeyboardInterrupt:
    print("Shutdown U96.")
    client.loop_stop()
    client.disconnect()
