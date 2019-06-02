import uRAD
from flask import Flask, render_template, request
import json

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
uTargets   = 1
uMaxRange  = 50
uMTI       = 0
uMovement  = 0

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0]
SNR       = [0]
movement  = [0]

api_dict = {}

# Switch on URAD
uRAD.turnON()

# Engine Loop
@app.route('/api/sensor/')
def SensorData():
    uRAD.detection(distances,0,SNR,0,0,movement)

    tof3.start_ranging(3)
    api_dict["back"]    = tof3.get_distance()
    tof3.stop_ranging()

    api_dict["distance"] = distances[0]
    api_dict["snr"] = SNR[0]
    api_dict["movement"] = movement[0]
    return(json.dumps(api_dict))

@app.route("/")
def home():
    return render_template("index.html")

@app.before_request
def option_autoreply():
    """ Always reply 200 on OPTIONS request """

    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()

        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']

        h = resp.headers

        # Allow the origin which made the XHR
        h['Access-Control-Allow-Origin'] = request.headers['Origin']
        # Allow the actual method
        h['Access-Control-Allow-Methods'] = request.headers['Access-Control-Request-Method']
        # Allow for 10 seconds
        h['Access-Control-Max-Age'] = "10"

        # We also keep current headers
        if headers is not None:
            h['Access-Control-Allow-Headers'] = headers

        return resp


@app.after_request
def set_allow_origin(resp):
    """ Set origin for GET, POST, PUT, DELETE requests """

    h = resp.headers

    # Allow crossdomain for other HTTP Verbs
    if request.method != 'OPTIONS' and 'Origin' in request.headers:
        h['Access-Control-Allow-Origin'] = request.headers['Origin']


    return resp

app.run(host='0.0.0.0')
