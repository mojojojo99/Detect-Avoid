import numpy as np
import matplotlib.pyplot as plt
Distances =  np.load("Distances.npy")
SNR = np.load("SNR.npy")
mA = np.load("movingAverage.npy")
F = np.load("est.npy")
t = np.arange(150)
plt.plot(t, Distances, 'r', t, F, 'b', t, mA, 'g')
plt.show()

