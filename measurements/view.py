import numpy as np
import matplotlib.pyplot as plt
Distances =  np.load("Distances.npy")
SNR = np.load("SNR.npy")
F = np.load("filter.npy")
mA = np.load("movingAverage.npy")
t = np.arange(150)
plt.plot(t, Distances, 'r', t, F, 'b', mA, 'g' )
plt.plot(SNR)
plt.show()

