import uRAD
from flask import Flask, render_template, request
import json

#app = Flask(__name__)

uMode      = 3
uFrequency = 5
uBW        = 240
uSamples   = 200
uTargets   = 5
uMaxRange  = 50
uMTI       = 0
uMovement  = 1

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0,0,0,0,0]
SNR = [0,0,0,0,0]
movement = [0]

api_dict = {}

# Switch on URAD
uRAD.turnON()

while True:
	past_distance = distances
	uRAD.detection(distances,0,SNR,0,0,movement)
	for idx in range(0,5):
		print(abs(past_distance[idx] - distances[idx]))
	api_dict["distance"] = distances[0]
	api_dict["snr"] = SNR[0]
	api_dict["movement"] = movement[0]

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
