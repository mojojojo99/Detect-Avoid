import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import cv2

count = 2001
while(count<2300):

    img = cv2.imread('frame'+str(count)+'.jpg', 0)
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

    plt.clf()

    count += 1
                                        # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
