import argparse
from collections import deque
from datetime import datetime
import json
import logging
import logging.config
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
from utils.framecounter import FrameCounter
from utils.videostream import VideoStream

lk_params = dict(winSize=(15, 15),
					maxLevel=2,
					criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

fb_params = dict(pyr_scale=0.5,
					levels=3,
					winsize=15,
					iterations=3,
					poly_n=5,
					poly_sigma=1.2,
					flags=0)

feature_params = dict(maxCorners=500,
						qualityLevel=0.3,
						minDistance=7,
						blockSize=7)

SHOW_SUBMASK = True
MASK_SIZE = 0.8  # proportion wrt main frame, any number <1.0

# for Lucas-Kanade
TRACK_LEN = 10
DETECT_INTERVAL = 5
MAX_TRACKING_PTS = 1000

# for Farneback
ACCUM_FLOW = True

def divergence(x, y, A):
	if len(np.shape(x)) is not 1:
		raise ValueError("x values should be a 1 dimensional array.")
	if len(np.shape(y)) is not 1:
		raise ValueError("y values should be a 1 dimensional array.")
	if np.shape(A) != (len(y), len(x)):
		raise ValueError("A should be a 2D array of size len(y) by len(x).")
	# preallocate output array
	d = np.zeros((len(y) - 1, len(x) - 1))

	# loop through rows and take the partial derivative of the y component
	# with respect to x
	for i, row in enumerate(A):
		if i is len(y) - 1:
			break
		d[i, :] += np.diff(np.real(row)) / np.diff(x)

	# loop through columns and take the partial derivative of the x component
	# with respect to y
	for i, col in enumerate(A.T):
		if i is len(x) - 1:
			break
		d[:, i] += np.diff(np.imag(col)) / np.diff(y)

	return d


def calc_divergence(flow):
	step = 4
	h, w = flow.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)
	fx, fy = flow[y, x].T

	width, height = fx.shape
	X = np.arange(0, width, 1)
	Y = np.arange(0, height, 1)
	A = np.zeros((height, width), dtype=np.complex128)
	A = fx.T + fy.T*1j
	D = divergence(X, Y, A)

	D_red = np.zeros((height-1, width-1, 3)).astype('uint8')
	D_red[D >= 0, 2] = D[D >= 0]*200
	D_red = cv.resize(
		D_red, None, fx=4, fy=4, interpolation=cv.INTER_LINEAR)
	cv.imshow('Divergence', D_red)


def warp_flow(img, flow):
	h, w = flow.shape[:2]
	flow = -flow
	flow[:, :, 0] += np.arange(w)
	flow[:, :, 1] += np.arange(h)[:, np.newaxis]
	res = cv.remap(img, flow, None, cv.INTER_LINEAR)
	return res


def draw_overlay(vis, opt_flow_algo, lines,
	show_optflow_lines, show_fps, show_track_count, show_submask,
	mask_bb, v_width, v_height, frame_counter, fps_text):
	if opt_flow_algo == "LK":
		for pts in lines:
			cv.circle(vis, (pts[-1, 0], pts[-1, 1]), 1, (0, 255, 0), -1)
	elif opt_flow_algo in ("FB", "DS") :
		for pts in lines:
			cv.circle(vis, (pts[0, 0], pts[0, 1]), 1, (0, 255, 0), -1)

	if show_optflow_lines:
		cv.polylines(vis, lines, False, (0, 0, 255))

	if show_track_count:
		x, y = 5, 40
		track_count_text = f'track count: {len(lines)} using {opt_flow_algo} algo'
		cv.putText(vis, track_count_text, (x, y), cv.FONT_HERSHEY_DUPLEX, 0.5,
				(0, 0, 0), thickness=2, lineType=cv.LINE_AA)
		cv.putText(vis, track_count_text, (x, y), cv.FONT_HERSHEY_DUPLEX, 0.5,
				(255, 255, 255), lineType=cv.LINE_AA)

	if show_fps:
		frame_counter.draw_text(vis, fps_text)

	if show_submask:
		cv.rectangle(vis, (mask_bb[0], mask_bb[1]), (mask_bb[2], mask_bb[3]), (0, 255, 0), 1)

	# draw centre crosshair
	ch_len = 5
	cv.line(vis, (v_width//2, v_height//2-ch_len), (v_width//2, v_height//2+ch_len), (255, 255, 255), 1)
	cv.line(vis, (v_width//2-ch_len, v_height//2), (v_width//2+ch_len, v_height//2), (255, 255, 255), 1)

	return vis


def calc_sparse_lines(tracks, gray_prev, gray, show_submask, show_optflow_lines, mask_bb):
	# use last elements in tracks as feature points (p0)
	p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
	# predict the next corresponding feature points (p1)
	p1, _st, _err = cv.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)
	# backward-check (p0r)
	p0r, _st, _err = cv.calcOpticalFlowPyrLK(gray, gray_prev, p1, None, **lk_params)
	# filter to select only good ones (p0 and p0r should be similar)
	d = abs(p0-p0r).reshape(-1, 2).max(-1)
	good = d < 1
	new_tracks = []
	for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
		if not good_flag:
			continue
		if show_submask:
			if x < mask_bb[0] or x > mask_bb[2]:
				continue
			if y < mask_bb[1] or y > mask_bb[3]:
				continue
		tr.append((x, y))
		if len(tr) > TRACK_LEN:
			del tr[0]
		new_tracks.append(tr)

	track_count = len(new_tracks)
	if track_count > MAX_TRACKING_PTS:
		new_tracks = new_tracks[track_count-MAX_TRACKING_PTS:]
	# lines = [np.int32(tr) for tr in new_tracks]
	lines = [np.int32((tr[0], tr[-1])) for tr in tracks]
	return new_tracks, lines


def calc_dense_lines(flow, tracks_fb, show_optflow_lines, step=8, accum_flow=False, offset=None):
	h, w = flow.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
	fx, fy = flow[y, x].T

	if accum_flow:
		tracks_fb.append((fx,fy))
		if len(tracks_fb) > 4:
			tracks_fb.popleft()
		for track in tracks_fb:
			fx = fx + track[0]
			fy = fy + track[1]

	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)

	if offset:
		lines = lines + np.array([offset, offset])
	return tracks_fb, lines


def parse_args(args):
	"""Parse the arguments."""
	parser = argparse.ArgumentParser(description='Tracks optical flow from video frame sequences')
	parser.add_argument('-c', '--camera', type=int, help='Id of webcam')
	parser.add_argument('-v', '--video_file', help='Video file path')
	parser.add_argument('-r', '--resize_video_height', type=int, default=0, help='Height of video to resize')
	parser.add_argument('-a', '--algorithm', default='LK', help='Optical flow algorithm. "LK" (Lucas Kanade)\
		by default if not specified, else use "FB" (Farneback) or "DIS" (Dense Inverse Search)')
	return parser.parse_args(args)


def main(args=None):
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	if not args.camera is None:
		src = args.camera
		cap = VideoStream(src, width=640, height=480, fps=30)
	elif args.video_file:
		src = args.video_file
		cap = VideoStream(src)
	else:
		logger.warning('No input source specified!')
		return

	o_width = cap.o_width
	o_height = cap.o_height
	o_fps = cap.o_fps
	logger.info(f'Original dimension - W: {o_width}, H: {o_height}, FPS: {o_fps:.2f}')

	opt_flow_algo = args.algorithm
	show_submask = SHOW_SUBMASK
	show_video_underneath = True
	show_second_video_win = False
	show_fps = True
	show_optflow_lines = True
	show_track_count = True
	frame_idx = 0
	scale = 1
	frame_counter = FrameCounter()
	frame_prev = cap.read()

	if args.resize_video_height:
		scale = args.resize_video_height/o_height
		v_width = int(o_width * scale)
		v_height = int(o_height * scale)
		logger.info(f'Expected resized - W: {v_width}, H: {v_height}')
		frame_prev = cv.resize(frame_prev, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
		v_height, v_width = frame_prev.shape[:2]
		logger.info(f'Actual resized - W: {v_width}, H: {v_height}')
	else:
		v_width, v_height = o_width, o_height

	gray_prev = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)

	# define center mask
	divisions = 1 / MASK_SIZE
	w_div = int(v_width//divisions)
	h_div = int(v_height//divisions)
	w_start = (v_width - w_div)//2
	h_start = (v_height - h_div)//2
	w_end = w_start + w_div
	h_end = h_start + h_div
	mask_bb = w_start, h_start, w_end, h_end

	flow = None
	lines = []
	tracks_lk = []
	tracks_fb = deque([])

	use_spatial_propagation = False
	use_temporal_propagation = True
	inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
	inst.setUseSpatialPropagation(use_spatial_propagation)

	fps_text = 'fps:000.0'
	if show_fps:
		frame_counter.setup_text(v_width, v_height, fps_text, margin=5)
	frame_counter.start()  # start fps counting before main loop
	start = datetime.now()

	while True:
		frame = cap.read()
		if frame is None: break
		if args.resize_video_height:
			frame = cv.resize(frame, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		if show_video_underneath:
			vis = frame.copy()
		else:
			vis = np.ones_like(frame) * 255

		if opt_flow_algo == "LK": # Lucas-Kanade
			if len(tracks_lk) > 0:
				tracks_lk, lines = calc_sparse_lines(tracks_lk, gray_prev, gray,
					show_submask, show_optflow_lines, mask_bb)
			if frame_idx % DETECT_INTERVAL == 0:  # find new features after every <DETECT_INTERVAL> frames
				mask = np.zeros_like(grayo)
				if show_submask:
					mask[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]] = 255
				else:
					mask[:] = 255
				# further masking to stop tracking areas around existing feature points
				for x, y in [np.int32(tr[-1]) for tr in tracks_lk]:
					cv.circle(mask, (x, y), 12, 0, -1)
				# cv.imshow('mask', mask)
				p = cv.goodFeaturesToTrack(gray, mask=mask, **feature_params) #  Shi-Tomasi Corner Detector
				if p is not None:
					for x, y in np.float32(p).reshape(-1, 2):
						tracks_lk.append([(x, y)])
		elif opt_flow_algo == "FB": # Farneback
			gray_tmp = gray
			gray_prev_tmp = gray_prev
			flow_pos_offset = None
			if show_submask:
				gray_tmp = gray[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]].copy()
				gray_prev_tmp = gray_prev[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]].copy()
				flow_pos_offset = mask_bb[0], mask_bb[1]
			flow = cv.calcOpticalFlowFarneback(gray_prev_tmp, gray_tmp, None, **fb_params)
			tracks_fb, lines = calc_dense_lines(
				flow, tracks_fb, show_optflow_lines, accum_flow=ACCUM_FLOW, offset=flow_pos_offset)
		elif opt_flow_algo == "DIS":
			gray_tmp = gray
			gray_prev_tmp = gray_prev
			flow_pos_offset = None
			if show_submask:
				gray_tmp = gray[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]].copy()
				gray_prev_tmp = gray_prev[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]].copy()
				flow_pos_offset = mask_bb[0], mask_bb[1]

			if flow is not None and use_temporal_propagation:
				# warp previous flow to get an initial approximation for the current flow:
				flow = inst.calc(gray_prev_tmp, gray_tmp, warp_flow(flow, flow))
			else:
				flow = inst.calc(gray_prev_tmp, gray_tmp, None)

			tracks_fb, lines = calc_dense_lines(
				flow, tracks_fb, show_optflow_lines, accum_flow=False, offset=flow_pos_offset)

		# calc_divergence(flow)

		vis = draw_overlay(vis, opt_flow_algo, lines,
			show_optflow_lines, show_fps, show_track_count,show_submask,
			mask_bb, v_width, v_height, frame_counter, fps_text)

		cv.imshow('Optical Flow', vis)
		if not show_video_underneath and show_second_video_win:
			cv.imshow('Input', frame)

		ch = cv.waitKey(1)
		if ch == 27:
			cap.halt()
			break
		elif ch == ord('l'):
			show_optflow_lines = not show_optflow_lines
		elif ch == ord('a'):
			if opt_flow_algo == "LK":
				opt_flow_algo = "FB"
			elif opt_flow_algo == "FB":
				opt_flow_algo = "DIS"
			elif opt_flow_algo == "DIS":
				opt_flow_algo = "LK"
		elif ch == ord('v'):
			show_video_underneath = not show_video_underneath
			if show_video_underneath:
				cv.destroyWindow('Input')
			show_second_video_win = not show_second_video_win
			if not show_video_underneath:
				cv.destroyWindow('Input')
		elif ch == ord('h'):
			show_second_video_win = not show_second_video_win
			if not show_video_underneath:
				cv.destroyWindow('Input')

		frame_idx += 1
		gray_prev = gray.copy()
		cur_fps = frame_counter.get_fps()  # update fps counting leaving main loop
		fps_text = f'fps:{cur_fps:5.1f}'

	logger.info(f'Total time elapsed: {(datetime.now()-start).total_seconds():.2f}seconds')
	cv.destroyAllWindows()


if __name__ == '__main__':
	# setup logging
	Path('logs').mkdir(parents=True, exist_ok=True)
	with open('json/logging_config.json', 'r') as log_config_file:
		config_dict = json.load(log_config_file)
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(__name__)

	main()
