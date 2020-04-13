"""
Image stitching using BRISK algorithm and knn Matching based on feature mapping
"""
import cv2
import numpy as np

#image 1
img_ = cv2.imread("left.jpg")
img_ = cv2.resize(img_, (500, 250), fx=0.5, fy=0.5)
img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

#image 2
img = cv2.imread("right.jpg")
img = cv2.resize(img, (500, 250), fx=0.5, fy=0.5)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#BRISK descriptor
descriptor = cv2.BRISK_create()

#detect features
kp1, des1 = descriptor.detectAndCompute(img1, None)
kp2, des2 = descriptor.detectAndCompute(img2, None)
cv2.imshow("original_image_left_keypoints", cv2.drawKeypoints(img_, kp1, None))
cv2.imshow("original_image_right_keypoints", cv2.drawKeypoints(img, kp2, None))
cv2.waitKey(0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

#feature mapping, vary paramter k for better results
k=0.03
good = []
for m, n in matches:
    if n.distance < k*n.distance:
        good.append(n)

draw_params = dict(matchColor = (0, 255, 0),
                        singlePointColor=None,
                        flags=2)

img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
cv2.imshow('stitched_image', img3)
cv2.waitKey(0)



cv2.destroyAllaWindows()
