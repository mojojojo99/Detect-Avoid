import numpy as np
import cv2 as cv
import argparse
#parser = argparse.argumentparser(description='this sample demonstrates lucas-kanade optical flow calculation. \
#                                                      the example file can be downloaded from: \
#                                                                                                    https://www.bogotobogo.com/python/opencv_python/images/mean_shift_tracking/slow_traffic_small.mp4')
#parser.add_argument('image', type=str, help='path to image file')
#args = parser.parse_args()
#cap = cv.videocapture(args.image)


# use webcam
cap = cv.videocapture(0)
# params for shitomasi corner detection
feature_params = dict( maxcorners = 100,
                               qualitylevel = 0.3,
                                                      mindistance = 7,
                                                                             blocksize = 7 )
# parameters for lucas kanade optical flow
lk_params = dict( winsize  = (15,15),
                          maxlevel = 2,
                                            criteria = (cv.term_criteria_eps | cv.term_criteria_count, 10, 0.03))
# create some random colors
color = np.random.randint(0,255,(100,3))
# take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtcolor(old_frame, cv.color_bgr2gray)
p0 = cv.goodfeaturestotrack(old_gray, mask = none, **feature_params)
# create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
        ret,frame = cap.read()
        frame_gray = cv.cvtcolor(frame, cv.color_bgr2gray)
        # calculate optical flow
        p1, st, err = cv.calcopticalflowpyrlk(old_gray, frame_gray, p0, none, **lk_params)
        # select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv.add(frame,mask)
            cv.imshow('frame',img)
            k = cv.waitkey(30) & 0xff
            if k == 27:
                break
                # now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)



cap.release()
cv2.destroyAllWindows()
