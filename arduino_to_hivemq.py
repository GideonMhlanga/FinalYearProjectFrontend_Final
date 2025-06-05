#!/usr/bin/env python3
import serial
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time

# Serial configuration (adjust port as needed)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

# HiveMQ configuration
MQTT_BROKER = "de52b19e5df54bcea7365ce5c2215ac2.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "N02019646a_finalyear/sensor_data" 
CLIENT_ID = "raspberrypi-gateway"
MQTT_USERNAME = "hivemq.webclient.1748702208336"
MQTT_PASSWORD = "BSrms06*G9JC;Dt1p,&a"

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
ser.flush()

# MQTT client setup
client = mqtt.Client(client_id=CLIENT_ID)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to HiveMQ broker")
    else:
        print(f"Connection failed with code {rc}")

def parse_arduino_data(raw_data):
    """Parse your Arduino's serial output format"""
    data = {}
    
    # Example for your format: "WIND,12.34,0.56,6.91|BATT,-1.23,85.0%"
    if '|' in raw_data:
        parts = raw_data.split('|')
        for part in parts:
            if part.startswith("WIND"):
                values = part.split(',')
                data.update({
                    'wind_voltage': float(values[1]),
                    'wind_current': float(values[2]),
                    'wind_power': float(values[3])
                })
            elif part.startswith("BATT"):
                values = part.split(',')
                data.update({
                    'battery_current': float(values[1]),
                    'battery_soc': float(values[2].replace('%', ''))
                })
    return data

def main():
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    
    print("Starting Arduino to HiveMQ bridge...")
    
    while True:
        try:
            if ser.in_waiting:
                raw_data = ser.readline().decode('utf-8').strip()
                print(f"Received from Arduino: {raw_data}")
                
                if raw_data:  # Only process if we got data
                    sensor_data = parse_arduino_data(raw_data)
                    
                    # Create MQTT payload
                    payload = {
                        'timestamp': datetime.utcnow().isoformat() + 'Z',
                        'device_id': 'arduino-uno-1',
                        'sensors': sensor_data
                    }
                    
                    # Publish to HiveMQ
                    client.publish(
                        topic=MQTT_TOPIC,
                        payload=json.dumps(payload),
                        qos=1
                    )
                    print(f"Published to HiveMQ: {payload}")
        
        except UnicodeDecodeError:
            print("Error decoding serial data - possibly corrupt")
        except ValueError as ve:
            print(f"Value error parsing data: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()