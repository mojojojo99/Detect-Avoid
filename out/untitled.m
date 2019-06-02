i = imread('/Users/jolim/Desktop/tree.jpg');
I = rgb2gray(i)
points = detectSURFFeatures(I);
imshow(I); hold on;
plot(points.selectStrongest(10));
