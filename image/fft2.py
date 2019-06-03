import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('far.png',0)
out = np.zeros(img.shape)
num = min(int(img.shape[0]/100), int(img.shape[1]/200))

# print(img.shape)
#  print(img)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# plt.imshow(magnitude_spectrum)
plt.contour(magnitude_spectrum)
plt.show()
