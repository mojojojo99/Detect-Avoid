print("Loading....")
import uRAD
import numpy as np
import RPi.GPIO as GPIO
import time

############## SET UP FOR RADAR ###########################

uMode      = 2
uFrequency = 5
uBW        = 5
uSamples   = 200
uTargets   = 1
uMaxRange  = 50
uMTI       = 0
uMovement  = 0

############## SET UP FOR TEST  #############################
resetDist = 1.0
numReadings = 20

# Initialze With Configurations
uRAD.loadConfiguration(uMode, uFrequency, uBW, uSamples, uTargets, uMaxRange, uMTI, uMovement)

# Storage Variables
distances = [0]
SNR = [0]
movement = [0]

# Switch on URAD
uRAD.turnON()

def Collect():
   angle = input("Angle of Reading: ")
   dist = input("Distance of Reading: ")
   print("----------------------------------------------")

   filename = "./data/" + angle + "_" + dist  + ".csv"
   dataDist = np.empty((0, uTargets *2))
   while True:
        uRAD.detection(distances,0,SNR,0,0,movement)

        if (distances[0] > resetDist):
            print("----------------------------------------------")
            print("Collecting")
            for i in range(numReadings):
                uRAD.detection(distances,0,SNR,0,0,movement)
                dataDist = np.vstack((dataDist, np.append(distances, SNR)))
                # dataSNR = np.vstack((dataSNR, SNR))
                print("Count: ", i)
                print("Distances: ", distances)
                print("SNR: ", SNR)
            break


        print(distances)
#        print(SNR)
    np.savetxt(filename, dataDist, delimiter=',', header="distance, SNR")

print("Ready! :-)")
while True:
    print("##########################################")
    Collect()


