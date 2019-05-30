#!/usr/bin/env python

import VL53L1X
import RPi.GPIO as GPIO
import time

XSHUT = 16

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(XSHUT,GPIO.OUT)
#GPIO.setup(VCC,GPIO.OUT)
#GPIO.output(VCC,GPIO.HIGH)
GPIO.output(XSHUT,GPIO.LOW)

tof1 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof1.open()
tof1.change_address(new_address = 0x2A)
#GPIO.output(XSHUT, GPIO.HIGH)
GPIO.setup(XSHUT,GPIO.IN)
tof1.open()
tof2 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof2.open()

#time.sleep(.5)

for i in range(9):
	tof1.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
	distance_in_mm = tof1.get_distance()    # Grab the range in mm
	print('Sensor:1',distance_in_mm)
	tof1.stop_ranging()                     # Stop ranging

#tof2 = VL53L1X.VL53L1X(i2c_bus=0, i2c_address=0x29)
#tof2.open()


	tof2.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
	distance_in_mm = tof2.get_distance()    # Grab the range in mm
	print('Sensor:2',distance_in_mm)
	tof2.stop_ranging()

GPIO.cleanup()
