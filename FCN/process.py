import os
import math
import cv2

directory = '../data_semantics/training/semantic/'
for filename in os.listdir(directory):
    name = os.path.join(directory, filename)
    img = cv2.imread(name,0)

    lower_reso = cv2.pyrDown(img)
    lower_reso = cv2.pyrDown(lower_reso)
    lower_reso = cv2.pyrDown(lower_reso)
    shape = lower_reso.shape
    for row in range(shape[0]):
        for col in range(shape[1]):
            p = lower_reso[row][col]
            check = int(p)
            if (check == 6 or check == 7 or check == 8 or check == 9 or check == 23):
                lower_reso[row][col] = 0
            else:
                lower_reso[row][col] = 1
    print (lower_reso)
    out_name = '../data_semantics/myTraining/semantic/' + filename
    cv2.imwrite(out_name,lower_reso)
    # cv2.imshow('img', lower_reso)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
