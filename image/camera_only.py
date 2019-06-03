import cv2 as cv
from datetime import datetime

from utils.framecounter import FrameCounter
from utils.videostream import VideoStream

# main loop variables
SHOW_WINDOW = True

def main():
	# generates video stream
	src, w, h, fps = 0, 640, 480, 30
	cap = VideoStream(src, width=w, height=h, fps=fps)

	# set dimensions for frames of the video stream
	width = cap.o_width
	height = cap.o_height
	src_fps = cap.o_fps
	frame_counter= FrameCounter()

	# starts fps before main loop
	frame_counter.start()
	start = datetime.now()

	# main loop
	while True:
		frame = cap.read()
		if SHOW_WINDOW:
			cv.imshow("CAMERA TEST", frame)
			k = cv.waitKey(1)
			if k == 27: # waits for ESC key to exit
				cap.halt()
				break

if __name__ == "__main__":
	main()
