"""
Image stitching using BRISK algorithm and knn Matching based on feature mapping
Use it as:
python video_stitching.py -v1 [video file 1 path] -v2 [video file 2 path]
"""
import cv2
import numpy as np
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-v1", "--path1", dest="v_file1", help="Path to Video 1.")
parser.add_option("-v2", "--path2", dest="v_file2", help="Path to Video 2.")

(options, args) = parser.parse_args()

video_file1 = options.v_file1
video_file2 = options.v_file2

descriptor = cv2.BRISK_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def img_transform(image, width, height):
    #img = cv2.imread(image)
    try:
        img = cv2.resize(image, (width, height))
        img_final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_final
    except Exception as e:
        print(str(e))

def video_cap(video_file1, video_file2):
    cap = cv2.VideoCapture(video_file1)
    cap2 = cv2.VideoCapture(video_file2)
    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        frame_trans = img_transform(frame, 320, 180)
        frame_trans2 = img_transform(frame2, 320, 180)
        kp1, des1 = descriptor.detectAndCompute(frame_trans, None)
        kp2, des2 = descriptor.detectAndCompute(frame_trans2, None)
        matches = bf.knnMatch(des1, des2, k=2)
        k = 0.03
        for m, n in matches:
            if n.distance < k*n.distance:
                good.append(n)

        draw_params = dict(matchColor = (0, 255, 0),
                                singlePointColor=None,
                                flags=2)

        img3 = cv2.drawMatches(frame1, kp1, frame2, kp2, good, None, **draw_params)
        cv2.imshow('stitched_image', img3)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


video_cap(video_file1, video_file2)
