import pupil_apriltags
import cv2
import os

def point(pt):
    return (int(pt[0]),int(pt[1]))

def draw_corners(img, detection):
    pt1 = point(detection.corners[0])
    cv2.circle(img, pt1,5,(0,255,0),-1)
    pt2 = point(detection.corners[1])
    cv2.line(img, pt1, pt2,(0,255,0),2)
    pt3 = point(detection.corners[3])
    cv2.line(img, pt1, pt3,(0,255,0),2)
    c = point(detection.center)
    cv2.circle(img, c,2,(0,0,255),-1)
#    cv2.putText(img,str(detection.tag_id), c, cv2.FONT_HERSHEY_PLAIN, 1.5,(0,0,255),2)

class AprilBoard:
    spacing = 0.1
    # 5x7 tag board
    tags = {
         0:[-3,-2,0],
         1:[-2,-2,0],
         2:[-1,-2,0],
         3:[0,-2,0],
         4:[1,-2,0],
         5:[2,-2,0],
         6:[3,-2,0],
        24:[-3,-1,0],
        25:[-2,-1,0],
        26:[-1,-1,0],
        27:[0,-1,0],
        28:[1,-1,0],
        29:[2,-1,0],
        30:[3,-1,0],
        48:[-3,0,0],
        49:[-2,0,0],
        50:[-1,0,0],
        51:[0,0,0],
        52:[1,0,0],
        53:[2,0,0],
        54:[3,0,0],
        72:[-3,1,0],
        73:[-2,1,0],
        74:[-1,1,0],
        75:[0,1,0],
        76:[1,1,0],
        77:[2,1,0],
        78:[3,1,0],
        96:[-3,2,0],
        97:[-2,2,0],
        98:[-1,2,0],
        99:[0,2,0],
        100:[1,2,0],
        101:[2,2,0],
        102:[3,2,0],
    }
    def __init__(self, spacing=0.1):
        self.spacing = spacing

    def draw(self, img):
        inc = 20
        for tag in self.tags:
            pt = self.tags[tag]
            cv2.putText(img, str(tag), (inc*pt[0]+inc*5, inc*pt[1]+5*inc),cv2.FONT_HERSHEY_PLAIN,0.5, (0,255,0))