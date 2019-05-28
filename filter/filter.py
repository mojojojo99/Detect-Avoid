import uRAD
import numpy as np
#app = Flask(__name__)

uMode      = 3
uFrequency = 5
uBW        = 240
uSamples   = 200
uTargets   = 5
uMaxRange  = 50
uMTI       = 0
uMovement  = 1

k =  0.75
# Write to file

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0,0,0,0,0]
SNR = [0,0,0,0,0]
all_distances = np.zeros(150)
all_SNR = np.zeros(150)
all_movingAverage = np.zeros(150)
all_est  = np.zeros(150)

movement = [0]
# Switch on URAD
uRAD.turnON()
count = 0
while count < 150:
    pastSNR =  SNR[0]
    pastDistance = distances[0]
    movingAverage = sum(all_distances[max(0, count - 5):count])
    if count != 0:
        movingAverage /= (count - max(0, count - 5))
    all_movingAverage[count]  = movingAverage


    all_distances[count] = distances[0]
    all_SNR[count] = SNR[0]
    uRAD.detection(distances,0,SNR,0,0,movement)


    if  pastSNR != 0 :
        gain = SNR[0]/pastSNR * k
        est = distances[0]*gain  +  (1 - gain) * pastDistance
    else:
        estDistance = distances[0]
    all_est[count] = estDistance
    print ("Count: " + str(count) )
    print (distances)
    print ("Moving Average: " + str(movingAverage))
    print ("Est: " + str(estDistance))

    count += 1
    #print (distances[0], end = " ", file=f)
    #print (SNR[0], file=f)
    np.save("SNR", all_SNR)
    np.save("Distances", all_distances)
    np.save("movingAverage", all_movingAverage)
    np.save("est", all_est)
