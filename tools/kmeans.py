import cv2
import time
import numpy as np
import sys
import imutils

from sklearn.cluster import KMeans


# cap = cv2.VideoCapture('inside2.m4v')
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('inside4.m4v')

fil = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (300,168))



i =0 


while(True):
    frame = cap.read()[1]
    
    if frame is None:
        break

    size_fudge = 120
    value_fudge = 0.5
    hue_fudge = 1000
    # resize the frame, blur it, and convert it to the HSV colour space
    frame = imutils.resize(frame, width = 300)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv_org = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    h, w, _ = hsv_org.shape
    print(h,w)
    hsv = hsv_org.astype(float)
    angle = 2*np.pi*(hsv[:,:,0] % 180)/180

    cx = hue_fudge * hsv[:,:,1] * hsv[:,:,2] * np.sin(angle)
    cy = hue_fudge * hsv[:,:,1] * hsv[:,:,2] * np.cos(angle)
    cz = hsv[:,:,2]*value_fudge
    cs = hsv[:,:,1]*value_fudge

    y,x = np.meshgrid(np.arange(w), np.arange(h))
    y = y*size_fudge/h
    x = x*size_fudge/w

    # print(cx.shape)
    # print(cy.shape)
    # print(cz.shape)
    # print(y.shape)
    # print(x.shape)
    out = np.stack((cx,cy,cz,cs,y,x),axis =2).reshape(h*w, 6)


    # print(out.shape)
    clusters = 10
    t = time.time()
    k = KMeans(n_clusters=clusters,precompute_distances=True).fit(out)
    print('done', time.time() -t)

    img = k.labels_.reshape(h,w).astype(np.uint8)
    print(img.shape)

    for i in range(clusters):
        ccx, ccy, ccz, ccs = k.cluster_centers_[i][:4]
        hue = (180*np.arctan2(ccx,ccy)/(2*np.pi)) % 180
        value = ccz/value_fudge
        saturation = ccs/value_fudge
        # saturation = np.sqrt( ccy**2 + ccx*2)/value
        # saturation = 255
        print(i, np.array([hue,saturation,value]))
        hsv_org[img == i] = np.array([hue,saturation,value]).astype(np.uint8)

    # edges = cv2.Canny(img*40,20,20)


    # hsv = np.ones([400,400,3]) * np.array([i%180, 255,255])
    # hsv = hsv.astype(np.uint8)
    # i +=1

    # print(hsv[0,0,:])

    show = cv2.cvtColor(hsv_org, cv2.COLOR_HSV2BGR)
    cv2.imshow('video', show)
    cv2.imshow('org', frame)
    fil.write(show)
    # cv2.imshow('video', img * 20)
    # cv2.imshow('edgled', edges)
    # cv2.imshow('video', hsv)
    # cv2.setMouseCallback('video',draw_circle)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('t'):
        mode = 'tennis'
    elif k == ord('b'):
        mode = 'bg'

cap.release()
cv2.destroyAllWindows()
