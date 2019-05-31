import cv2
import numpy as np
import sys
from __future__ import division
import picamera

from UDPComms import Subscriber, Publisher

from fetch import find_ball_direct

capture_res = (2592, 1944)


cam_params = {}


def main(argv):
    idx_camera = argv[0]
    offset_degrees = argv[1]

    reference = Subscriber(9010)
    detection_results = Publisher(902 + idx_camera)

    camera = picamera.PiCamera(sensor_mode=2, resolution=capture_res, framerate=10)
    while True:
        try:
            cam_params = reference.get()
        except UDPComms.timeout:
            if "range_hue" not in cam_params:
                continue

        camera.shutter = cam_params["shutter"]
        camera.iso = cam_params["iso"]

        image = np.empty((capture_res[1] * capture_res[0] * 3), dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((capture_res[1], capture_res[0], 3))

        hsv_ranges = (cam_params["range_hue"], cam_params["range_sat"], cam_params["range_val"])

        heading, distance = find_ball_direct(image, hsv_ranges, offset_degrees)

        if(distance > 0):
            result = {"heading":heading, "distance":distance}
            classification.send(result)


if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Usage: pifetch [camera index] [heading offset degrees]")
    main(sys.argv[1:])