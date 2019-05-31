import cv2
import numpy as np
import sys
from __future__ import division
import picamera

from UDPComms import Publisher

from fetch import find_ball_direct

capture_res = (2592, 1944)

shutter = 4
iso = 400
iso_step = 100
ISO_MIN = 100
ISO_MAX = 6400

cam_params = {"shutter" : shutter}

def main(argv):

    reference = Publisher(9010)

    camera = picamera.PiCamera(sensor_mode=2, resolution=capture_res, framerate=10)

    while True:

        image = np.empty((capture_res[1] * capture_res[0] * 3), dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((capture_res[1], capture_res[0], 3))

        truth = image[ 
            int(img_h * 0):int(img_h * 0.3), 
            int(img_w * 0):int(img_w * 0.3), 
        :]   
        truth_hsv = cv2.cvtColor(truth, cv2.COLOR_BGR2HSV)

        brightpoint = np.percentile(truth_hsv[:, :, 2], 95)
        toobright = brightpoint > 180
        toodark = brightpoint < 80

        if(toodark or toobright):
            if toobright:
                iso -= iso_step

            if toodark:
                iso += iso_step

            iso = max(ISO_MIN, min(iso, ISO_MAX))
            camera.iso = iso
            cam_params["iso"] = iso

            print(f"ISO changed: {iso}")
            continue

        # exposure is ok, let's send over reference colors
        cam_params["range_hue"] = np.percentile(truth_hsv[:, :, 0], (5, 95))
        cam_params["range_sat"] = np.percentile(truth_hsv[:, :, 1], (5, 95))
        cam_params["range_val"] = np.percentile(truth_hsv[:, :, 2], (5, 95))
        reference.send(cam_params)

        print(cam_params)



if __name__ == '__main__':
    if(len(sys.argv) > 1):
        print("Usage: pireference (no options)")
    main(sys.argv[1:])
    camera.close()