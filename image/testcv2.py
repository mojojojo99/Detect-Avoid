import cv2
img = cv2.imread('./frames/frame0.jpg',1)
print(img.shape)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
