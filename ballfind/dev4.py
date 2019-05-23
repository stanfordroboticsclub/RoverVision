import cv2
import numpy as np
import sys
import imutils


cap = cv2.VideoCapture('inside2.m4v')

f = open("CalibrateResults/results.txt",'r')
data = f.readlines()
values = []
for i in range(0,len(data)):
    values.append(int(data[i].strip()))

#calibrate hsv values first using range finder script
low_thresh = (values[0], values[1], values[2]) #current: (22, 12, 0)
high_thresh = (values[3], values[4], values[5]) #current: (56, 188, 255)

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
    hsv4 = cv2.GaussianBlur(frame, (9,9), 0)

    # construct a mask for the ball's signature, then perform a series of dilations and
    # erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, low_thresh, high_thresh)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #frame_annotated = frame
    
    params = cv2.SimpleBlobDetector_Params()
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, \
    # dp=2, minDist=50, param1=100, param2=5, minRadius=10, maxRadius=100)
    params.minThreshold = 1
    
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 100000
    
    params.filterByCircularity = True
    params.minCircularity = 0.05
    
    params.filterByConvexity = True
    params.minConvexity = .05
    
    params.filterByInertia = .05
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    reversemask = 255 - mask
    keypoints = detector.detect(reversemask)

    counter = 0
    for x in keypoints:
        if counter == 0:
            diameter = int(x.size)
            xval = (int(x.pt[0])   - (diameter / 2))
            yval = (int(x.pt[1]) - (diameter / 2))
            cv2.rectangle(hsv4, (int(xval), int(yval)), (int(xval) + diameter, int(yval) + diameter) , (0, 0, 255), 2)
            counter += 1


    cv2.imshow('video', hsv4)
    if cv2.waitKey(1)==27: # esc Key
        break

cap.release()
cv2.destroyAllWindows()