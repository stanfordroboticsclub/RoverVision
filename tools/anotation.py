import cv2
import numpy as np
import sys
import imutils


# cap = cv2.VideoCapture('inside2.m4v')
cap = cv2.VideoCapture(0)

# f = open("CalibrateResults/results.txt",'r')
# data = f.readlines()
# values = []
# for i in range(0,len(data)):
#     values.append(int(data[i].strip()))

#calibrate hsv values first using range finder script
# low_thresh = (values[0], values[1], values[2]) #current: (22, 12, 0)
# high_thresh = (values[3], values[4], values[5]) #current: (56, 188, 255)

low_thresh = (22, 12, 0)
ball = (40, 150, 200)
# low_thresh = (22, 32, 20)
high_thresh = (56, 188, 255)

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

print("Ball low threshold: " + str(low_thresh))
print("Ball high threshold: " + str(high_thresh))

mouseX,mouseY = 0,0
mode = 'tennis'
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    global hsv4
    # if event == cv2.EVENT_LBUTTONDBLCLK:
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        print(mouseX,mouseY, "hsv", mode, hsv4[y,x,:])



data = [["bg", [ 99, 116, 149]],
["bg", [ 37,  95, 105]],
["bg", [34, 90, 98]],
["bg", [ 36,  94, 107]],
["bg", [16, 61, 68]],
["bg", [44, 58, 78]],
["bg", [123, 193, 255]],
["bg", [145,  95,  47]],
["bg", [135,  78,  41]],
["bg", [230, 162, 107]],
["bg", [174, 124,  98]],
["bg", [ 52,  73, 123]],
["bg", [ 76, 107, 113]],
["bg", [122, 128, 138]],
["bg", [ 87, 110, 119]],
["bg", [ 91, 130, 213]],
["bg", [124, 177, 254]],
["bg", [ 69,  98, 146]],
["bg", [157, 100,  52]],
["tennis", [ 83, 248, 255]],
["tennis", [ 39, 142, 166]],
["tennis", [255, 255, 255]],
["tennis", [255, 255, 255]],
["tennis", [139, 255, 255]],
["tennis", [ 44, 152, 175]],
["tennis", [188, 255, 255]],
["tennis", [ 51, 159, 185]],
["tennis", [ 24, 112, 135]],
["tennis", [119, 255, 255]],
["tennis", [242, 255, 255]],
["tennis", [245, 255, 255]],
["tennis", [255, 255, 255]],
["tennis", [ 42, 150, 172]],
["tennis", [100, 253, 255]],
["tennis", [192, 255, 255]],
["tennis", [ 80, 234, 247]],
["tennis", [214, 255, 255]],
["tennis", [ 56, 136, 143]],
["tennis", [ 77, 221, 238]],
["tennis", [124, 255, 255]],
["tennis", [ 75, 200, 197]]]


def dist(image):
    i_max = np.ones_like(image)
    dis_max = float('inf') * (np.ones_like(image).astype(float))[:,:,0]
    for col in data:
        d = np.linalg.norm(image - np.array(col[1]), axis=2)
        mask = d < dis_max
        dis_max[mask]  = d[mask]
        if col[0] == 'tennis':
            i_max[mask] = 255
        elif col[0] == 'bg':
            i_max[mask] = 0
    return i_max





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
    # mask = cv2.inRange(hsv, low_thresh, high_thresh)
    # mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = dist(hsv4)
    mask = cv2.erode(mask, None, iterations=5)
    mask = cv2.dilate(mask, None, iterations=5)
    
    #frame_annotated = frame
    
    params = cv2.SimpleBlobDetector_Params()
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, \
    # dp=2, minDist=50, param1=100, param2=5, minRadius=10, maxRadius=100)
    params.minThreshold = 1
    
    params.filterByArea = True
    # params.minArea = 4
    # params.maxArea = 100000

    params.minArea = 100
    params.maxArea = 20000
    
    params.filterByCircularity = True
    # params.minCircularity = 0.05
    params.minCircularity = 0.6
    
    params.filterByConvexity = True
    params.minConvexity = .2
    
    params.filterByInertia = .7
    
    detector = cv2.SimpleBlobDetector_create(params)
    
    reversemask = 255 - mask
    keypoints = detector.detect(reversemask)

    counter = 0
    for x in keypoints:
        if counter < 5:
            diameter = int(x.size)
            xval = (int(x.pt[0])   - (diameter / 2))
            yval = (int(x.pt[1]) - (diameter / 2))
            cv2.rectangle(hsv4, (int(xval), int(yval)), (int(xval) + diameter, int(yval) + diameter) , (0, 0, 255), 2)
            counter += 1


    m = np.linalg.norm(frame - ball, axis=2)

    m = m/np.max(m) * 255

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(np.max(m), np.min(m))
    m = m.astype(np.uint8)
    # print(gray.shape, type(gray))
    # print(np.max(gray), np.min(gray))

    # print(m.shape, type(m))
    # print(np.max(m), np.min(m))

    # cv2.imshow('video', m)
    cv2.imshow('mask', mask)
    cv2.imshow('video', hsv4)
    cv2.setMouseCallback('video',draw_circle)
    # if cv2.waitKey(1)==27: # esc Key
    #     break

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('t'):
        mode = 'tennis'
    elif k == ord('b'):
        mode = 'bg'

cap.release()
cv2.destroyAllWindows()
