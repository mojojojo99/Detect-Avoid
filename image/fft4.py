import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('far.png',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.imshow(magnitude_spectrum)
plt.show()

