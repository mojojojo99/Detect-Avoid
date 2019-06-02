mexOpenCV detectORBFeatures.cpp
% Then call this function in MATLAB just like any other MATLAB command
im = imread("cameraman.tif"); 
keypoints = detectORBFeatures(im);