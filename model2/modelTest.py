import uRAD
from flask import Flask, render_template, request
import json
import numpy as np
import VL53L1X
import RPi.GPIO as GPIO
import time


app = Flask(__name__)

################## SETUP GPIO and TOF ##################
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

################## SETUP GPIO and TOF ##################

uMode      = 2
uFrequency = 5
uBW        = 240
uSamples   = 200
uTargets   = 5
uMaxRange  = 50
uMTI       = 0
uMovement  = 0

################# SETUP MODEL #########################

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0, 0, 0, 0, 0]
SNR = [0, 0, 0, 0, 0]
movement = [0]

# Switch on URAD
uRAD.turnON()
count = 0

data = np.zeros(15)
while True:
    # Sensing
    uRAD.detection(distances,0,SNR,0,0,movement)
    # start = time.time()
    tof3.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
    distance_in_mm = tof3.get_distance()    # Grab the range in mm
    tof3.stop_ranging()                     # Stop ranging
    # end = time.time()
    # print(end - start)
    print(distances)
    print(distance_in_mm)







    count += 1
