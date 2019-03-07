import cv2
import numpy as np
import sys

cap = cv2.VideoCapture('autodrive.m4v')


while(True):
    frame = cap.read()[1]

    #### Start


    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    candidate_mask_1 = cv2.inRange(frame_hsv, (359, 0, 0), (360, 255, 150))
    # candidate_mask_2 = cv2.inRange(frame_hsv, (0, 150, 20), (360, 250, 100))
    candidate_mask_2 = cv2.inRange(frame_hsv, (0, 180, 50), (360, 240, 100))

    candidate_mask = cv2.bitwise_or(candidate_mask_1, candidate_mask_2)
    candidate_mask = cv2.blur(candidate_mask, (10,10))


    frame_annotated = frame

    circles = cv2.HoughCircles(candidate_mask, cv2.HOUGH_GRADIENT, \
        dp=2, minDist=50, param1=100, param2=5, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0,:]:
            # draw bounding circle
            cv2.circle(frame_annotated, (x, y), r, (0,255,0), 2)
            # plot a point at centroid
            cv2.circle(frame_annotated, (x, y), 3, (0,0,255), 3)


    #### END

    cv2.imshow('video', frame_annotated)
    if cv2.waitKey(1)==27: # esc Key
        break

cap.release()
cv2.destroyAllWindows()