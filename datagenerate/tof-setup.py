#!/usr/bin/env python

import VL53L1X
import RPi.GPIO as GPIO
import time

TOF_GPIO = [4, 23, 24]

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Turn Off ALL Pins
for gpio in TOF_GPIO:
	GPIO.setup(gpio,  GPIO.OUT)
	GPIO.output(gpio, GPIO.LOW)


# Turn On First TOF Sensor
GPIO.output(TOF_GPIO[0], GPIO.HIGH)
tof1 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof1.open()
tof1.change_address(new_address = 0x2A)

GPIO.output(TOF_GPIO[1], GPIO.HIGH)
tof2 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof2.open()
tof2.change_address(new_address = 0x2B)

GPIO.output(TOF_GPIO[2], GPIO.HIGH)
tof3 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof3.open()


for i in range(25):
	tof3.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
	distance_in_mm = tof3.get_distance()    # Grab the range in mm
	print('Sensor:1',distance_in_mm)
	tof3.stop_ranging()                     # Stop ranging


GPIO.cleanup()
