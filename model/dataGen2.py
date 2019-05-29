import sys, termios, tty, os, time
import uRAD
import numpy as np


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

while True:

    # Sensing
    uRAD.detection(distances,0,SNR,0,0,movement)
    all_distances.append(distances[0])
    all_SNR.append(SNR[0])

    tof3.start_ranging(3)                   # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
    distance_in_mm = tof3.get_distance()    # Grab the range in mm
    tof3.stop_ranging()                     # Stop ranging
    all_TOF.append(distance_in_mm)
    print (distance_in_mm)
    print (distances[0])

    char = getch()

    if (char == "p"):
        print("Stop!")
        exit(0)

    if (char == "a"):
        print("Left pressed")
        time.sleep(button_delay)


np.save("SNR", all_SNR)
np.save("Distances", all_distances)
np.save("Labels", all_labels)

GPIO.cleanup()
