import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('near.png',0)
out = np.zeros(img.shape)
num = min(int(img.shape[0]/100), int(img.shape[1]/200))
for n in range(num):
    win = img[n*100:(n+1)*100]
    win = win[:][n*200:(n+1)*200]
    print(win.shape)


# print(img.shape)
#  print(img)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
