
import cv2
import os
import numpy as np
import json
import time
import pupil_apriltags
from AprilBoard import *
import sys
import argparse


import inspect
import builtins

def custom_print(*args, **kwargs):
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    builtins.print(f"{os.path.basename(caller.filename)}:{caller.lineno} - ", end="")
    builtins.print(*args, **kwargs)

# Override the built-in print function
print = custom_print

def point(pt):
    return (int(pt[0]),int(pt[1]))


def draw_detection(img, detection, color=(255,0,0)):
    pt1 = point(detection.corners[0])
    cv2.circle(img, pt1,5,color,-1)
    pt2 = point(detection.corners[1])
    cv2.line(img, pt1, pt2,color,2)
    pt3 = point(detection.corners[3])
    cv2.line(img, pt1, pt3,color,2)
    c = point(detection.center)
    cv2.circle(img, c,2,color,-1)
    cv2.putText(img,str(detection.tag_id), c, cv2.FONT_HERSHEY_PLAIN, 1.5,color,2)


def crop_image(image, bbox_str):
    """
    Crops an image based on a bounding box string format '[top:bottom,left:right]'.

    Args:
        image: A NumPy array representing the image.
        bbox_str: A string representing the bounding box in format '[top:bottom,left:right]'.

    Returns:
        A NumPy array representing the cropped image.
    """
    try:
        slices = bbox_str.strip('[]').split(',')
        top, bottom = map(int, slices[0].split(':'))
        left, right = map(int, slices[1].split(':'))
    except ValueError:
        raise ValueError('Invalid bounding box format. Use [top:bottom,left:right]')

    # Crop the image
    return image[top:bottom, left:right]

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="use fisheye calibration mode",action="store_true", default='.')
parser.add_argument("--cameraID", help="subtring filter for files in a directory for a specific camera",default=None)
parser.add_argument("--name", help="string name for camera",default='Unamed')
parser.add_argument("--fisheye", help="use fisheye calibration mode",action="store_true", default=False)
parser.add_argument("--crop", help="crop images to this size", default=None)
parser.add_argument("--flip", help="flip images horizontal", action="store_true", default=False)
parser.add_argument("--show", help="show images with opencv", action="store_true", default=False)
parser.add_argument("--unwarp", help="save unwarpped images", action="store_true", default=False)
args = parser.parse_args()

fisheyeMode = args.fisheye
path = args.path
cameraID = args.cameraID
crop = args.crop

print(f"Searching path: {path}")

allfiles = os.listdir(path)
images = []
for f in allfiles:
    #skip rereading unwarped images
    if "unwarped" in f:
        continue
    if cameraID is not None:
        if cameraID in f and ".jpg" in f:
            images.append(f)
    elif ".jpg" in f:
        images.append(f)


if len(images) == 0:
    print("No images found")
    exit()

board = AprilBoard()

detector = pupil_apriltags.Detector()
objectPtsAll = []
imagePtsAll = []

# load image matches
for path in images:
    print(path)
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if crop is not None:
        img = crop_image(img, crop)
        print(img.shape)
    if args.flip:
        img = cv2.flip(img, 1)

    result = detector.detect(img)

    imgPts = []
    objPts = []

    print(f"\tFound {len(result)} tags in image")
    for detection in result:
        draw_detection(img, detection)
        imgPts.append(detection.center)        
        objPts.append(board.tags[detection.tag_id])
    if len(objPts) >= 6:
        # objectPtsAll.append(np.array(objPts,dtype=np.float32))
        # imagePtsAll.append(np.array(imgPts,dtype=np.float32))

        objPts = np.array(objPts, dtype=np.float32).reshape(1,-1,3)
        imgPts = np.array(imgPts, dtype=np.float32).reshape(1,-1,2)
        # print(objPts)
        # break
        objectPtsAll.append(objPts)
        imagePtsAll.append(imgPts)

K = np.identity(3)
D = np.zeros(4)

if len(objectPtsAll) == 0:
    print("No images with AprilBoard Detections")
    exit()

if fisheyeMode:
    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPtsAll, 
        imagePtsAll,
        (img.shape[1], img.shape[0]), K,D)
else:
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objectPtsAll, 
        imagePtsAll,
        (img.shape[1], img.shape[0]), K,D)


result = {
        "fisheye": fisheyeMode,
        "flip": args.flip,
        "crop": crop,
        "matrix": K.tolist(),
        "distortion": D.tolist()
        }
print(json.dumps(result, indent=2, sort_keys=True))
print(f"Used {len(objectPtsAll)}/{len(images)} images with reprojection Error {ret}")

# save calibration to file
outfile = f"intrinsics_{args.name}_{int(time.time())}.json"
with open(outfile,"w") as fp:
    json.dump(result, fp)
    print(f"Saved to {outfile}")

# show undistorted images
for path in images:
    img = cv2.imread(path)
    if crop is not None:
        img = crop_image(img, crop)
    if args.flip:
        img = cv2.flip(img, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detector.detect(gray)

    for detection in result:
        draw_detection(img, detection,(0,255,0))
    if fisheyeMode:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (img.shape[1], img.shape[0]), cv2.CV_16SC2)
        img2 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        img2 = cv2.undistort(img, K, D)    
    if args.show:
        cv2.imshow(cameraID, img2)
        c = cv2.waitKey()&0xFF
        if c == 27:
            break
    if args.unwarp:
        out_path = f"unwarped_{args.name}_{path}"
        cv2.imwrite(out_path, img2)
        print(f"Saved unwarped to {out_path}")
