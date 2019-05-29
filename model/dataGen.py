import uRAD
import numpy as np
import sys, termios, tty, os, time

import VL53L1X
import RPi.GPIO as GPIO


TOF_GPIO = [4, 23, 24]

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Turn Off ALL Pins
for gpio in TOF_GPIO:
	GPIO.setup(gpio,  GPIO.OUT)
	GPIO.output(gpio, GPIO.LOW)

GPIO.output(TOF_GPIO[2], GPIO.HIGH)
tof3 = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
tof3.open()


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

button_delay = 0.2

#uRad Settings
uMode      = 3
uFrequency = 5
uBW        = 240
uSamples   = 200
uTargets   = 1
uMaxRange  = 50
uMTI       = 0
uMovement  = 1

# Write to file

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0]
SNR = [0]
all_distances = []
all_SNR = []
all_TOF = []
all_labels = []

movement = [0]

# Switch on URAD
uRAD.turnON()
count = 0

while True:
    

    char = getch()

    if (char == "p"):
        print("Stop!")
        exit(0)
    elif (char == "a"):
        print("Left pressed")
        time.sleep(button_delay)

    count += 1

np.save("SNR", all_SNR)
np.save("Distances", all_distances)
np.save("Labels", all_labels)

GPIO.cleanup()
