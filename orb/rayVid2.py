import numpy as np
import quaternion as qt
import scipy.optimize as optimize
from scipy.optimize import Bounds
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

def readFile():

    file = open("./images/airsim_rec.txt","r")

    params = np.empty((0,7))

    images = []

    for line in file:
        fields = line.split()
        images.append(fields[8]) #append image file name
        params = np.vstack((params, np.array(fields[1:8])))

    return params.astype(np.float), images


def orbDetect (file1, file2):


    img2 = cv2.imread('./images/' + file1, 0)
    img1 = cv2.imread('./images/' + file2, 0)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    # Match descriptors.
    # matches = bf.match(des1,des2)
    matches = bf.knnMatch(des1,des2, k=2)

    # Sort them in the order of their distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

    # filtered = matches[:10]

    good = []
    pts1 = []
    pts2 = []

    for (m,n) in matches:
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, img1, img2

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

    return  np.array(trans), rot


def convertPoints(pts1, pts2):

    pts3 = np.subtract(pts1, [1280/2, 720/2])
    pts4 = np.subtract(pts2, [1280/2, 720/2])

    pts3 = -pts3
    pts4 = -pts4
    return pts3, pts4

def staged(params_prev, params_cur):

    ## Staged Calculatations
    # 3d line expressed in terms of possible z values, f is the fov
    alpha = 90  #fov in degrees
    # scalex = 360/np.tan(alpha*(np.pi)/360)
    # scaley = 640/np.tan(alpha*(np.pi)/360)
    scalex = 360/np.tan(alpha*(np.pi)/360)
    scaley = 640/np.tan(alpha*(np.pi)/360)

    trans, rot = process([np.array(params_prev),
                          np.array(params_cur)])

    return scalex, scaley, trans, rot

def line(x, y):
    return np.array([x/scalex, y/scaley, 1])

def lineTransform(l):
    return np.matmul(rot, l)

def dist(cur_line, prev_line):
    return lambda ds: np.linalg.norm(ds[0]*cur_line - lineTransform(prev_line)*ds[1])


def getDepth(pts3, pts4):
    zs=[]

    for i in range(len(pts3)):
        cur_line = line(pts3[i][0], pts3[i][1])
        prev_line  = lineTransform(line(pts4[i][0], pts4[i][1]))

        #CurLine- PrevLine
        AB = np.zeros((3,3))
        AB[:, 0] = cur_line
        AB[:, 1] = -prev_line
        AB[:, 2] = trans

        firsteq = np.dot(np.transpose(cur_line), AB)
        secondeq = np.dot(np.transpose(prev_line), AB)
        b =  np.array([-firsteq[2], -secondeq[2]])
        firsteq = firsteq[:2]
        secondeq = secondeq[:2]


        x = np.linalg.solve(np.array([firsteq, secondeq]), -b)


        zs.append(x[0])

    return zs

def drawpts(img1, pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for pt1 in pts1:
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    return img1

############################ MAIN #############################################

params, images = readFile()
print (params.shape)

for i in range(len(images)-1):
    pts1, pts2, img1, img2 = orbDetect(images[i], images[i+1])
    pts3, pts4 = convertPoints(pts1, pts2)
    scalex, scaley, trans, rot = staged(params[i], params[i+1])

    try:
            # your code that will (maybe) throw
        zs = getDepth(pts3, pts4)
    except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print(err)
            else:
                raise

    fig,ax = plt.subplots(1)
    # img3 = drawpts(img1,pts1)
    # ax.imshow(img3)
    outImg = np.zeros(img1.shape)

    for z , (x, y) in zip (zs, pts1):
        outImg[y][x] = z
        print (z , ' ', x , ' ', y)

    # for z, (x, y) in zip(zs, pts1):
        # ax.annotate(z, (x, y))
    print(outImg)
    ax.imshow(outImg)

    plt.savefig('hello' + str(i) + '.png')


