import os
import paho.mqtt.client as mqtt
import ssl

U96_SUBSCRIBE_TOPIC = "/playerBall"
U96_PUBLISH_TOPIC = "/opponentBall"

_HERE = os.path.dirname(__file__)
CA_CERT = os.path.join(_HERE, "certs", "ca.crt")
CLIENT_CERT = os.path.join(_HERE, "certs", "u96-client.crt")
CLIENT_KEY = os.path.join(_HERE, "certs", "u96-client.key")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"Connected with result code {rc}")
    client.subscribe(U96_SUBSCRIBE_TOPIC)

def on_message(client, userdata, msg):
    msg = str(msg.payload.decode('utf-8'))
    print(f"Received message from {U96_SUBSCRIBE_TOPIC}: {msg}")

broker_address = "127.0.0.1"
broker_port = 8883

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=CLIENT_KEY
)
client.tls_insecure_set(True)
client.connect(broker_address, broker_port)
client.publish(U96_PUBLISH_TOPIC, "from u96")
client.loop_forever()
