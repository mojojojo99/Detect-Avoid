import numpy as np
import matplotlib.pyplot as plt

p = np.load('./predict.npy')
y = np.load('./label.npy')
thresholded = np.zeros(p.shape)
for i in range(p.size):
    if p[i] >= 0.5:
        thresholded[i] = 1


plt.plot(thresholded[:100])
plt.plot(y[:100])
plt.title('Prediction')
plt.legend(['predicted', 'actual'])
plt.show()
