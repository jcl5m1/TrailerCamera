import apriltag
import cv2
import os
import numpy as np
import json
import time
from AprilBoard import *

dir = "trailer_sim"
match = "back"
allfiles = os.listdir(dir)
images = []
calibration_file = None
for f in allfiles:
    if match in f and ".jpg" in f:
        images.append(os.path.join(dir,f))
    if match in f and "intrinsics" in f:
        calibration_file = os.path.join(dir,f)
 
board = AprilBoard()

detector = apriltag.Detector()
print(calibration_file)
K = np.identity(3)
D = np.zeros(4)
with open(calibration_file,'r') as fp:
    K,D = json.load(fp)
    K = np.array(K)
    D = np.array(D)

# load images
for path in images:
    print(path)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    result = detector.detect(img)

    #rectified params
    rectifiedPts = []
    rectifiedScale = 30
    M = np.array([
        [rectifiedScale,0,img.shape[1]/2],
        [0,rectifiedScale,img.shape[0]/2],
        [0,0,1]
    ])

    imgPts = []
    objPts = []
    for detection in result:
        #draw_corners(img, detection)
        imgPts.append(detection.center)        
        objPts.append(board.tags[detection.tag_id])
        rp = np.array(board.tags[detection.tag_id])
        rp[2] = 1
        rp = np.matmul(M, rp)
        rectifiedPts.append(rp[:2])
    if len(imgPts) < 4:
        continue
    imgPts = np.array(imgPts)
    objPts = np.array(objPts) 
    rectifiedPts = np.array(rectifiedPts)

    #lens undistort image and points
    img2 = cv2.undistort(img, K, D)
    undistored_points = cv2.undistortPoints(imgPts,K,D)

    #homography
    src_points = []
    for pt in undistored_points:
        pt = np.matmul(K, np.array([pt[0][0],pt[0][1],1]))
        src_points.append(pt)
    src_points = np.array(src_points)
    H, mask = cv2.findHomography(src_points,rectifiedPts)
    img3 = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
    cv2.imshow("rectified", img3)
    cv2.imshow("image", img) 
    cv2.moveWindow("rectified", img.shape[1],0)
    c = cv2.waitKey() & 0xFF
    if c == 27:
        break