import smbus
from smbus2 import SMBus
import time

I2C_Bus = SMBus(1) # Replace 0 with 1 for Raspberry Pi 3
SLAVE_ADD = 0x08    # i2c Address of the Arduino

def get_ArduinoData():
    return I2C_Bus.read_i2c_block_data(SLAVE_ADD, 0, 31) 

while True:
	ArduinoData = get_ArduinoData()
	print(ArduinoData)
	time.sleep(1)
