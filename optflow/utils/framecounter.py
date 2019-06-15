import cv2 as cv


class FrameCounter:
	def __init__(self):
		self._last_tick = None
		self._freq = cv.getTickFrequency()
		self._latest_fps = 1
		self._frame_num = 0
		self._font_pos = 0, 0
		self.font_type = cv.FONT_HERSHEY_DUPLEX
		self.font_scale = 0.5
		self.font_thickness_inner = {'lineType': cv.LINE_AA, 'thickness': 1}
		self.font_thickness_outer = {'lineType': cv.LINE_AA, 'thickness': 2}

	def start(self):
		self._last_tick = cv.getTickCount()

	def get_fps(self):
		cur_tick = cv.getTickCount()
		duration = (cur_tick-self._last_tick) / self._freq
		self._frame_num += 1

		if duration > 1:
			self._latest_fps = self._frame_num / duration
			self._last_tick = cur_tick
			self._frame_num = 0

		return self._latest_fps

	def setup_text(self, win_width, win_height, text, margin):
		(label_width, label_height), baseline = cv.getTextSize(text, self.font_type,
															   self.font_scale, 2)
		self._font_pos = (margin,
						  label_height + baseline)

	def draw_text(self, frame, fps_text):
		cv.putText(frame, fps_text,
				   self._font_pos, self.font_type, self.font_scale,
				   (0, 0, 0), **self.font_thickness_outer)
		cv.putText(frame, fps_text,
				   self._font_pos, self.font_type, self.font_scale,
				   (255, 255, 255), **self.font_thickness_inner)
