import cv2
import numpy as np
import sys
import imutils


cap = cv2.VideoCapture('inside6.m4v')

f = open("CalibrateResults/results.txt",'r')
data = f.readlines()
values = []
for i in range(0,len(data)):
    values.append(int(data[i].strip()))

low_thresh = (values[0], values[1], values[2]) # (22, 12, 0)
high_thresh = (values[3], values[4], values[5]) # (56, 188, 255)

print("Ball low threshold: " + str(low_thresh))
print("Ball high threshold: " + str(high_thresh))

while(True):
    frame = cap.read()[1]
    
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV colour space
    frame = imutils.resize(frame, width = 600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the ball's signature, then perform a series of dilations and
    # erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, low_thresh, high_thresh)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    frame_annotated = frame
    
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, \
                               dp=2, minDist=50, param1=100, param2=5, minRadius=10, maxRadius=100)
        
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles.shape)
        circles = np.squeeze(circles, axis=0)
        numCircles = min(circles.shape[0], 1)
        print(numCircles)
        circles_pruned = circles[:numCircles,:]
        for x, y, r in circles_pruned:
            # draw bounding circle
            cv2.circle(frame_annotated, (x, y), r, (0,255,0), 2)
            # plot a point at centroid
            cv2.circle(frame_annotated, (x, y), 3, (0,0,255), 3)
                                                               
    cv2.imshow('video', frame_annotated)
    if cv2.waitKey(1)==27: # esc Key
        break

cap.release()
cv2.destroyAllWindows()