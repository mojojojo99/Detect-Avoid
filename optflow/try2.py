import numpy as np
import cv2 as cv
import argparse

# params for shitomasi corner detection
feature_params = dict( maxCorners = 100,  qualityLevel = 0.3,  minDistance = 7,  blockSize = 7 )
# parameters for lucas kanade optical flow
lk_params = dict( winSize = (15,15),
                          maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# create some random colors
color = np.random.randint(0,255,(100,3))
# take first frame and find corners in it
old_frame = cv.imread('./images/img__0_1559877619886931400.png', 1)
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
frame= cv.imread('./images/img__0_1559877620242687500.png', 1)
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# create a mask image for drawing purposes
mask = np.zeros_like(old_gray)

# calculate optical flow
p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# select good points
good_new = p1[st==1]
good_old = p0[st==1]
# draw the tracks
for i,(new,old) in enumerate(zip(good_new, good_old)):
    a,b = new.ravel()
    c,d = old.ravel()
    mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    frame = cv.circle(frame_gray,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
cv.imshow('frame',img)
# now update the previous frame and previous points
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1,1,2)

cv.waitKey(0)
cv.destroyAllWindows()

