import cv2
import numpy as np
import sys

# cap = cv2.VideoCapture('autodrive.m4v')
img = cv2.imread("test3.jpg")
truth = cv2.imread("sample3.png")




img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
truth_hsv = cv2.cvtColor(truth, cv2.COLOR_BGR2HSV)

# truth_hsv = cv2.blur(truth_hsv, (10,10))
# img_sample = cv2.cvtColor(truth_hsv, cv2.COLOR_HSV2BGR)
# cv2.imshow('sample', img_sample)
# cv2.waitKey(0)

print(truth_hsv.shape)


# img_rt= cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
# (25, 75)

target_hue = np.percentile(truth_hsv[:, :, 0], (5, 95))
# mean_hue = np.mean(truth_hsv[0])
# target_hue = [mean_hue*0.9, mean_hue*1.1]

# target_sat = [0, 255]
target_sat = np.percentile(truth_hsv[:, :, 1], (5, 95))
# mean_sat = np.mean(truth_hsv[1])
# target_sat = [mean_sat*0.9, mean_sat*1.1]

# target_val = [0, 255]
target_val = np.percentile(truth_hsv[:, :, 2], (5, 95))

print(target_hue, target_sat, target_val)


candidate_mask = cv2.inRange(img_hsv, 
    (target_hue[0], target_sat[0], target_val[0]), 
    (target_hue[1], target_sat[1], target_val[1])
)

# kernel = np.ones((10,10), np.uint8) 
# candidate_mask = cv2.erode(candidate_mask, kernel)
# candidate_mask = cv2.dilate(candidate_mask, kernel)


# candidate_mask_2 = cv2.inRange(frame_hsv, (0, 150, 20), (360, 250, 100))
# candidate_mask_2 = cv2.inRange(frame_hsv, (0, 180, 50), (360, 240, 100))

# candidate_mask = cv2.bitwise_or(candidate_mask_1, candidate_mask_2)
# candidate_mask = cv2.blur(candidate_mask, (10,10))


# frame_annotated = img

# circles = cv2.HoughCircles(candidate_mask, cv2.HOUGH_GRADIENT, \
#     dp=2, minDist=50, param1=100, param2=5, minRadius=1, maxRadius=100)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for x, y, r in circles[0,:]:
#         # draw bounding circle
#         cv2.circle(frame_annotated, (x, y), r, (0,255,0), 2)
#         # plot a point at centroid
#         cv2.circle(frame_annotated, (x, y), 3, (0,0,255), 3)

cv2.imshow('annot', cv2.resize(candidate_mask, (1280, 800)))
cv2.imshow('orig', cv2.resize(img, (1280, 800)))
cv2.waitKey(0)
cv2.destroyAllWindows()

#### END
