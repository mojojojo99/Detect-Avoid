import uRAD
from flask import Flask, render_template, request
import numpy as np
import VL53L1X
import RPi.GPIO as GPIO
import time
import socket

################SETUP SOCKET############################
TCP_IP = '169.254.75.176' # this IP of my pc. When I want raspberry pi 2`s as a client, I replace it with its IP '169.254.54.195'
TCP_PORT = 5005
BUFFER_SIZE = 1024
MESSAGE  = ""

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

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
uTargets   = 1
uMaxRange  = 50
uMTI       = 0
uMovement  = 0

################# SETUP MODEL #########################


# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0]
SNR = [0]
movement = [0]

# Switch on URAD
uRAD.turnON()
count = 0

data = np.zeros(15)
while True:

    # Sensing
    uRAD.detection(distances,0,SNR,0,0,movement)

    tof3.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
    distance_in_mm = tof3.get_distance()    # Grab the range in mm
    tof3.stop_ranging()                     # Stop ranging


    if count >= 4:
        data[:12] = data[3:15]
        data[12:15] = np.array([distances[0], SNR[0], distance_in_mm])
        MESSAGE = ""
        for i in data:
            MESSAGE +=  str(i) + " "
        # res = model.predict(data)
        # print (res)

        s.sendall(MESSAGE.encode('utf-8'))

    else:
        data[count:count+3]  = np.array([distances[0], SNR[0], distance_in_mm])




    count += 1

s.close()
