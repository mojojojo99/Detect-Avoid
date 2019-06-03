import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

cap = cv2.VideoCapture('trimmed_vid.mp4')

def grab_frame():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    return magnitude_spectrum, img

def update(i):
    mag, img = grabframe()

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.contour(magnitude_spectrum)
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])


im1 = ax1.imshow(grab_frame())

                def update(i):
                        im1.set_data(grab_frame())

                        ani = FuncAnimation(plt.gcf(), update, interval=200)
                        plt.show()
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# plt.imshow(magnitude_spectrum)
plt.contour(magnitude_spectrum)
plt.show()
while(True):

