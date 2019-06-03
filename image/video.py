import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import cv2

cap = cv2.VideoCapture('trimmed_vid.mp4')
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    count += 1

    if count > 140:
        img = np.zeros(0)
        f = np.zeros(0)
        fshift=np.zeros(0)
        magnitude_spectrum = np.zeros(0)
        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.contour(magnitude_spectrum)
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

        plt.savefig('testplot'+ str(count)+'.png')
        out = cv2.imread('testplot'+ str(count)+'.png')
        cv2.imshow('spectrum', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

                                        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
