#!/usr/bin/env python

import VL53L1X
import RPi.GPIO as GPIO
import time

#VCC = 18
XSHUT = 16

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(XSHUT,GPIO.OUT)
GPIO.output(XSHUT,GPIO.LOW)

tof1 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof1.open()
tof1.change_address(new_address = 0x2B)

GPIO.cleanup()
