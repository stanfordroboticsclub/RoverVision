import sys
import numpy as np
from camUnit import camUnit

RES_WIDTH = 3280 #horizontal resolution of pi camera = 3280 pixels
AOV = 62 #angle of view of pi camera about 62 degree
OBJ_WIDTH = 0.063 #tennis ball diameter = 63 mm

cam1 = camUnit(RES_WIDTH, AOV, OBJ_WIDTH, -15, 0.5)
cam2 = camUnit(RES_WIDTH, AOV, OBJ_WIDTH, 0, 0.5)
cam3 = camUnit(RES_WIDTH, AOV, OBJ_WIDTH, 15, 0.5)

def get_average_distance():
    sum = 0
    if cam1.has_obj() == True:
        sum += cam1.get_distance()
    if cam2.has_obj() == True:
        sum += cam2.get_distance()
    if cam3.has_obj() == True:
        sum += cam3.get_distance()
    return sum / 3

def get_average_heading():
    sum = 0
    if cam1.has_obj() == True:
        sum += cam1.get_heading()
    if cam2.has_obj() == True:
        sum += cam2.get_heading()
    if cam3.has_obj() == True:
        sum += cam3.get_heading()
    return (sum / 3) * (180/np.pi)

def main():
    print(get_average_distance())
    print(get_average_heading())

if __name__ == "__main__":
    main()
