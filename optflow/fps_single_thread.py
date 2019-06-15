from datetime import datetime
import cv2 as cv
import numpy as np
from utils.framecounter import FrameCounter
import os

CONVERT_BW = False
RESIZE_FRAME = False
OUTPUT_FRAME_HEIGHT = 360
SHOW_WINDOW = True


def main():
	src = 1
	cap = cv.VideoCapture(src)
	w, h, fps = 640, 480, 30
	cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
	cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
	cap.set(cv.CAP_PROP_FPS, fps)
	width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
	src_fps = cap.get(cv.CAP_PROP_FPS)
	frame_counter = FrameCounter()
	scale = 1
	cur_fps = 0.0
	fps_text = 'fps:000.0'
	text_margin = 5

	print(f'Original width: {width}\nOriginal height: {height}\nOriginal fps: {src_fps}\n')

	if RESIZE_FRAME:
		scale = OUTPUT_FRAME_HEIGHT / height
		width = int(width * scale)
		height = int(height * scale)
		print(f'New width: {width}\nNew height: {height}')

	if SHOW_WINDOW:
		frame_counter.setup_text(width, height, fps_text, text_margin)

	start = datetime.now()
	frame_counter.start()  # start fps counting before main loop
	while True:
		ret, frame = cap.read()

		if not ret:
			break

		if CONVERT_BW:
			frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		if RESIZE_FRAME:
			frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
		if SHOW_WINDOW:
			frame_counter.draw_text(frame, fps_text)
			cv.imshow("fps_single_thread", frame)
			k = cv.waitKey(1)
			if k == 27:  # wait for ESC key to exit
				break
		else:
			print(fps_text)

		cur_fps = frame_counter.get_fps()  # update fps counting leaving main loop
		fps_text = f'fps:{cur_fps:5.1f}'

	print(f'Total elapsed: {(datetime.now()-start).total_seconds():.2f}seconds')
	cap.release()
	if SHOW_WINDOW:
		cv.destroyAllWindows()


if __name__ == "__main__":
	main()
