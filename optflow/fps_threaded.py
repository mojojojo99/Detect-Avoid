from datetime import datetime
import time
import cv2 as cv
import numpy as np
import os

from utils.framecounter import FrameCounter
from utils.videostream import VideoStream
from threading import Thread, current_thread

CONVERT_BW = False
RESIZE_FRAME = False
OUTPUT_FRAME_HEIGHT = 360
SHOW_WINDOW = True


def main():
	src = 0
	# cap = VideoStream(src)
	# w, h, fps = 320, 240, 30
	w, h, fps = 640, 480, 30
	# w, h, fps = 1280, 960, 30
	# w, h, fps = 1920, 1440, 30
	cap = VideoStream(src, width=w, height=h, fps=fps)

	width = cap.o_width
	height = cap.o_height
	src_fps = cap.o_fps
	frame_counter= FrameCounter()
	scale = 1
	cur_fps = 0.0
	fps_text = 'fps:000.0'
	text_margin = 5

	print(f'Original width: {width}\nOriginal height: {height}\nOriginal fps: {src_fps}\n')

	if RESIZE_FRAME:
		scale = OUTPUT_FRAME_HEIGHT/height
		width = int(width * scale)
		height = int(height * scale)
		print(f'New width: {width}\nNew height: {height}')

	if SHOW_WINDOW:
		frame_counter.setup_text(width, height, fps_text, text_margin)

	frame_counter.start() # start fps counting before main loop
	start = datetime.now()
	while True:
		frame = cap.read()
		if frame is None:
			cap.halt()
			break

		if CONVERT_BW:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		if RESIZE_FRAME:
			frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
		if SHOW_WINDOW:
			frame_counter.draw_text(frame, fps_text)
			cv.imshow("fps_threaded", frame)
			k = cv.waitKey(1)
			if k == 27: # wait for ESC key to exit
				cap.halt()
				break
		else:
			print(fps_text)

		cur_fps = frame_counter.get_fps()  # update fps counting leaving main loop
		fps_text = f'fps:{cur_fps:5.1f}'

	print(f'Total elapsed: {(datetime.now()-start).total_seconds():.2f}seconds')
	if SHOW_WINDOW:
		cv.destroyAllWindows()


if __name__ == "__main__":
	main()
