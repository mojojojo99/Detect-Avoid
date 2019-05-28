import numpy as np
import matplotlib.pyplot as plt
Distances =  np.load("Distances.npy")
SNR = np.load("SNR.npy")
t = np.arange(150)
plt.plot(t, Distances, 'r', t, SNR, 'b')
plt.plot(SNR)
plt.show()

