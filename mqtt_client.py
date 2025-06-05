import paho.mqtt.client as mqtt
import json
from datetime import datetime
import configparser
import ssl
from database import db  # Import your database manager

# Load configuration
config = configparser.ConfigParser()
config.read('/etc/arduino_mqtt.conf')

# Configuration
MQTT_BROKER = config['hivemq']['broker']
MQTT_PORT = int(config['hivemq']['port'])
ARDUINO_TOPIC = "N02019646a_finalyear/arduino_data"
PZEM_TOPIC = "N02019646a_finalyear/pzem_solardata"
CLIENT_ID = "dashboard-client"
MQTT_USERNAME = config['hivemq']['username']
MQTT_PASSWORD = config['hivemq']['password']

class MQTTDataHandler:
    def __init__(self):
        self.client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS)
        
        # Set up callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        # Connect to broker
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_start()
    
    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("Connected to MQTT broker")
            client.subscribe([(ARDUINO_TOPIC, 1), (PZEM_TOPIC, 1)])
        else:
            print(f"Connection failed with code {reason_code}")
    
    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        print(f"Disconnected from MQTT broker with code {reason_code}")
        if reason_code != 0:
            client.reconnect()
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            timestamp = datetime.now().isoformat()
            payload['mqtt_timestamp'] = timestamp
            
            if msg.topic == ARDUINO_TOPIC:
                # Process Arduino data
                self.process_arduino_data(payload)
            elif msg.topic == PZEM_TOPIC:
                # Process PZEM data
                self.process_pzem_data(payload)
                
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def process_arduino_data(self, data):
        """Process and store Arduino data"""
        try:
            # Store wind data
            if all(k in data for k in ['wind_voltage', 'wind_current', 'wind_power']):
                db.store_wind_data(
                    voltage=data['wind_voltage'],
                    current=data['wind_current'],
                    power=data['wind_power'],
                    timestamp=data.get('timestamp')
                )
            
            # Store battery data
            if all(k in data for k in ['battery_current', 'battery_soc']):
                db.store_battery_data(
                    current=data['battery_current'],
                    soc=data['battery_soc'],
                    timestamp=data.get('timestamp')
                )
                
        except Exception as e:
            print(f"Error storing Arduino data: {e}")
    
    def process_pzem_data(self, data):
        """Process and store PZEM solar data"""
        try:
            if 'voltage' in data and 'current' in data and 'power' in data:
                db.store_solar_data(
                    voltage=data['voltage'],
                    current=data['current'],
                    power=data['power'],
                    timestamp=data.get('timestamp')
                )
        except Exception as e:
            print(f"Error storing PZEM data: {e}")

# Initialize MQTT client when module is imported
mqtt_handler = MQTTDataHandler()