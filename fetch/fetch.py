import cv2
import numpy as np
import sys
from __future__ import division


# keepout region defined as list of integer indices, ordered as (y_min, y_max, x_min, x_max), using top-left origin
def find_ball(img, truth, keepout, mount_offset_deg, hfov = 68):

    img_h, img_w = img.shape[0:2]
    vfov = hfov * img_h / img_w

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    truth_hsv = cv2.cvtColor(truth, cv2.COLOR_BGR2HSV)
    truth_hsv = cv2.blur(truth_hsv, (10,10))

    img_sample = cv2.cvtColor(truth_hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('sample', img_sample)
    cv2.waitKey(0)


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

    # zero out keepout zone
    candidate_mask[keepout[0]:keepout[1], keepout[2]:keepout[3]] = 0


    kernel = np.ones((10,10), np.uint8) 
    candidate_mask = cv2.dilate(candidate_mask, kernel)
    candidate_mask = cv2.erode(candidate_mask, kernel)


    hot_indices = np.transpose(np.nonzero(candidate_mask))
    c, r = cv2.minEnclosingCircle(hot_indices)

    # reject ball candidates that are larger than 10pct of horizontal FoV
    # if(r > 0.1*img_h):
    #     print("FP suppressed")
    #     return img, candidate_mask

    prior_r_cm = 3.2
    baseline_cm = (img_w / r) * prior_r_cm
    distance_cm = (baseline / 2) / math.tan(math.radians(hfov/2))

    # reject ball candidates closer than 1m
    if(distance < 100):
        print("FP suppressed")
        return 0, -1

    c_px = (int(c[0]), int(c[1]))
    img_annot = cv2.circle(img, c_px, int(r), (0,0,255), thickness=3)

    # get normalized x position with respect to hfov, in range [-0.5, 0.5]
    screen_coord_x = (c[1] - (img_w / 2))/img_w
    # multiply by known fov to get 
    heading = (screen_coord_x * hfov) + mount_offset_deg

    return heading, distance
    # return img_annot, candidate_mask


def find_ball_direct(img, hsv_ranges, mount_offset_deg, hfov = 68):

    img_h, img_w = img.shape[0:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    target_hue, target_sat, target_val = hsv_ranges

    candidate_mask = cv2.inRange(img_hsv, 
        (target_hue[0], target_sat[0], target_val[0]), 
        (target_hue[1], target_sat[1], target_val[1])
    )


    kernel = np.ones((10,10), np.uint8) 
    candidate_mask = cv2.dilate(candidate_mask, kernel)
    candidate_mask = cv2.erode(candidate_mask, kernel)

    hot_indices = np.transpose(np.nonzero(candidate_mask))
    c, r = cv2.minEnclosingCircle(hot_indices)

    prior_r_cm = 3.2
    baseline_cm = (img_w / r) * prior_r_cm
    distance_cm = (baseline / 2) / math.tan(math.radians(hfov/2))

    # reject ball candidates closer than 1m
    if(distance < 100):
        print("FP suppressed")
        return 0, -1

    c_px = (int(c[0]), int(c[1]))
    img_annot = cv2.circle(img, c_px, int(r), (0,0,255), thickness=3)

    # get normalized x position with respect to hfov, in range [-0.5, 0.5]
    screen_coord_x = (c[1] - (img_w / 2))/img_w
    # multiply by known fov to get 
    heading = (screen_coord_x * hfov) + mount_offset_deg

    return heading, distance
    # return img_annot, candidate_mask



def display_img(name, img, max_height=1000):
    scale_factor = 1
    img_h, img_w = img.shape[0:2]
    if(img_h > max_height):
        scale_factor = max_height/img_h

    cv2.imshow(name, cv2.resize(img, 
        (int(img_w * scale_factor), int(img_h * scale_factor))
    ))


def main(argv):
    path_input = argv[0]

    # cap = cv2.VideoCapture('autodrive.m4v')
    img = cv2.imread(path_input)
    img_h, img_w, _ = img.shape

    # truth = img[ 
    #     int(img_h * 0.8):int(img_h * 0.9), 
    #     int(img_w * 0.3):int(img_w * 0.7), 
    # :]

    truth = img[ 
        int(img_h * 0.85):int(img_h * .95), 
        int(img_w * 0.4):int(img_w * 0.6), 
    :]

    keepout = [
        int(img_h*0.6), int(img_h*1),
        int(img_w*0), int(img_w*1),
    ]

    keepout_drawn = cv2.rectangle(img, (keepout[2], keepout[0]), (keepout[3], keepout[1]), (0, 0, 255), thickness=3)

    # display_img("keepout region", keepout_drawn)
    # cv2.waitKey(0)

    annot, candidate_mask = find_ball(img, truth, keepout)

    # candidate_mask = cv2.blur(candidate_mask, (10,10))

    # frame_annotated = img

    
    display_img("mask", candidate_mask)
    display_img("annot", annot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
