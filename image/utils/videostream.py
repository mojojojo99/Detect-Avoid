from threading import Thread
import cv2 as cv
import time


class VideoStream:
	def __init__(self, src, width=None, height=None, fps=None, delay=0):
		self.stream = cv.VideoCapture(src)

		if width:
			self.stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
		if height:
			self.stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)
		if fps:
			self.stream.set(cv.CAP_PROP_FPS, fps)

		self.o_width = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
		self.o_height = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
		self.o_fps = int(self.stream.get(cv.CAP_PROP_FPS))
		self.grabbed, self.frame = self.stream.read()
		self.stop = False
		self.delay = delay
		Thread(target=self.refresh, args=()).start()

	def refresh(self):
		while not self.stop:
			if self.grabbed:
				self.grabbed, self.frame = self.stream.read()
			else:
				self.halt()
			time.sleep(self.delay)
		self.stream.release()

	def read(self):
		return self.frame

	def halt(self):
		self.stop = True
