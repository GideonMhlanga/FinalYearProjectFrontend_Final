import minimalmodbus
import serial
import time

# Configure PZEM-017 (Modbus RTU)
instrument = minimalmodbus.Instrument('/dev/ttyUSB0', slaveaddress=1)
instrument.serial.baudrate = 9600
instrument.serial.timeout = 0.5

# Serial port to Arduino (USB)
arduino_serial = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

try:
    while True:
        # Read PZEM-017 DC values
        voltage = instrument.read_register(0, 1)  # Voltage (0.1V)
        current = instrument.read_register(1, 2)   # Current (0.01A)
        power = instrument.read_register(3, 1)     # Power (0.1W)

        # Format: "SOLAR,<voltage>,<current>,<power>"
        data_str = f"SOLAR,{voltage},{current},{power}\n"
        
        # Send to Arduino
        arduino_serial.write(data_str.encode())
        print(f"Sent to Arduino: {data_str.strip()}")

        time.sleep(5)  # Match Arduino's 5-second delay

except KeyboardInterrupt:
    arduino_serial.close()
    print("Stopped.")