import uRAD
import sys
#app = Flask(__name__)

uMode      = 3
uFrequency = 5
uBW        = 240
uSamples   = 200
uTargets   = 5
uMaxRange  = 1
uMTI       = 0
uMovement  = 1

# Write to file
f = open("output.txt", "a")

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0,0,0,0,0]
SNR = [0,0,0,0,0]
movement = [0]

api_dict = {}
# Switch on URAD
uRAD.turnON()
count = 0
while True:
    count += 1
    past_distance = distances
    uRAD.detection(distances,0,SNR,0,0,movement)
    print ("Count: " + str(count) )
    print (distances)
    print (distances[0], end = " ", file=f)
    print (SNR[0], file=f)
f.close()
