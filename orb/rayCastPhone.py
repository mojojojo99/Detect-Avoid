import numpy as np
import quaternion as qt
import scipy.optimize as optimize
from scipy.optimize import Bounds
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

file = open("./images/airsim_rec.txt","r")

for line in file:
      #Let's split the line into an array called "fields" using the ";" as a separator:
      fields = line.split(" ")

img2 = cv2.imread('./phone/1.jpg', 0)
img1 = cv2.imread('./phone/2.jpg', 0)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

filtered = matches[:10]

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for m in filtered:
    good.append(m)
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

def process(params):
    prev = params[0]
    cur = params[1]
    trans = [0,0,0]
    trans[:2] = -cur[1:3] + prev[1:3]
    trans[2] = -cur[0] + prev[0]

    currot = [0,0,0,0]
    prevrot = [0,0,0,0]
    currot[0] = cur[3]
    currot[3] = cur[4]
    currot[1:3] = cur[5:7]
    prevrot[0] = prev[3]
    prevrot[3] = prev[4]
    prevrot[1:3] = prev[5:7]
    prevrot = qt.as_rotation_matrix(qt.as_quat_array(prevrot))
    currot = qt.as_rotation_matrix(qt.as_quat_array(currot))

    rot = np.matmul(currot, np.linalg.inv(prevrot))
    print (trans)
    return  np.array(trans), rot

## Staged Calculatations
# 3d line expressed in terms of possible z values, f is the fov
alpha = 90  #fov in degrees
scalex = 1512/np.tan(alpha*(np.pi)/360)
scaley = 2016/np.tan(alpha*(np.pi)/360)
trans, rot = process([np.array([0, 0.000000, 0, 0.998629, 0.000000, 0, 0.015]),
                      np.array([1, 14, 0.0, 0.999391, 0.000000, 0.0, 0.03])])
def line(x, y):
    return np.array([x/scalex, y/scaley, 1])

def lineTransform(l):
    return np.matmul(rot, l)

def dist(cur_line, prev_line):
    return lambda ds: np.linalg.norm(ds[0]*cur_line - lineTransform(prev_line)*ds[1])
pts3 = np.subtract(pts1, [3024/2, 4032/2])
pts4 = np.subtract(pts2, [3024/2, 4032/2])

pts3 = -pts3
pts4 = -pts4
zs=[]

for i in range(len(pts3)):
    cur_line = line(pts3[i][0], pts3[i][1])
    prev_line  = lineTransform(line(pts4[i][0], pts4[i][1]))

    print (cur_line)
    print (prev_line)
    #CurLine- PrevLine
    AB = np.zeros((3,3))
    AB[:, 0] = cur_line
    AB[:, 1] = -prev_line
    AB[:, 2] = -trans

    firsteq = np.dot(np.transpose(cur_line), AB)
    secondeq = np.dot(np.transpose(prev_line), AB)
    b =  np.array([-firsteq[2], -secondeq[2]])
    firsteq = firsteq[:2]
    secondeq = secondeq[:2]


    x = np.linalg.solve(np.array([firsteq, secondeq]), -b)
    # initial_guess =  30*np.random.random_sample((2,)) + 10
    # result = optimize.least_squares(dist(cur_line, prev_line), initial_guess, bounds=(0, np.inf))
    # result = optimize.minimize(dist(cur_line, prev_line), initial_guess, method = 'Nelder-Mead', bounds=bounds)
    # result = optimize.minimize(dist(cur_line, prev_line), initial_guess)
    # print (result)
    # ds = result.x
    # zs.append(ds[0])
    print(firsteq)
    print(secondeq)
    print(b)
    print("Xs: ", x)
    print (x[0]*cur_line)
    print (x[1]*prev_line + trans)
    zs.append(x[0])

print(zs)
def drawpts(img1, pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    return img1

fig,ax = plt.subplots(1)

img3 = drawpts(img1,pts1)
ax.imshow(img3)

for z, (x, y) in zip(zs, pts1):
    ax.annotate(z, (x, y))

plt.show()
