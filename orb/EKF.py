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

def getAbsoluteScale(i, params)	
  x = params[i][0]
  y = params[i][1]
  z = params[i][2]
  if i = 0:
    return sqrt(x*x + y*y + z*z)
  else:
    x_prev = params[i-1][0]
    y_prev = params[i-1][1]
    z_prev = params[i-1][2]
    return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev))


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

def recoverPose():

  focal = 640
  pp = (640, 360)
  F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, 1.0, 1.0)


  Mat prevImage = img_2;
  Mat currImage;
  vector<Point2f> prevFeatures = points2;
  vector<Point2f> currFeatures;

  char filename[100];

  R_f = R.clone();
  t_f = t.clone();

  clock_t begin = clock();

  namedWindow( "Road facing camera", WINDOW_AUTOSIZE );// Create a window for display.
  namedWindow( "Trajectory", WINDOW_AUTOSIZE );// Create a window for display.

  Mat traj = Mat::zeros(600, 600, CV_8UC3);

  for(int numFrame=2; numFrame < MAX_FRAME; numFrame++)	{
  	sprintf(filename, "/home/avisingh/Datasets/KITTI_VO/00/image_2/%06d.png", numFrame);
    //cout << numFrame << endl;
  	Mat currImage_c = imread(filename);
  	cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
  	vector<uchar> status;
  	featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

  	E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
  	recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

    Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


   for(int i=0;i<prevFeatures.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    }

  	scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

    //cout << "Scale is " << scale << endl;

    if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

      t_f = t_f + scale*(R_f*t);
      R_f = R*R_f;

    }
  	
    else {
     //cout << "scale below 0.1, or incorrect translation" << endl;
    }
    
   // lines for printing results
   // myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << endl;

  // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
 	  if (prevFeatures.size() < MIN_NUM_FEAT)	{
      //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
      //cout << "trigerring redection" << endl;
 		  featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

 	  }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;

    int x = int(t_f.at<double>(0)) + 300;
    int y = int(t_f.at<double>(2)) + 100;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    imshow( "Road facing camera", currImage_c );
    imshow( "Trajectory", traj );

    waitKey(1);

  }

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Total time taken: " << elapsed_secs << "s" << endl;

  //cout << R_f << endl;
  //cout << t_f << endl;

  return 0;
}