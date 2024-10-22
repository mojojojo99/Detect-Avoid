import uRAD
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

count = 0 
while True:
	count += 1
	past_distance = distances
	uRAD.detection(distances,0,SNR,0,0,movement)
	if (count % 10 == 0): 
		print (distances)
