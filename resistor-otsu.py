#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

global hsv_img

img = cv2.imread("template.png")
img = cv2.copyMakeBorder(img, 5,5,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
colour_img = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("blur", blur)
cv2.imshow("otsu", th3)

edges = cv2.Canny(th3, 50, 200)
cv2.imshow("edges", edges)

contour_img = img.copy()
contours, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print cv2.contourArea(sorted_contours[0])

cv2.drawContours(contour_img, sorted_contours, 0, (0,255,0), 3)
cv2.imshow("contours", contour_img)

peri = cv2.arcLength(sorted_contours[0], True)
approx = cv2.approxPolyDP(sorted_contours[0], 0.02 * peri, True)

mask = np.zeros(img.shape,np.uint8)
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(sorted_contours[0], cv2.cv.CV_DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y) + 15
righty = int(((cols-x)*vy/vx)+y) + 15
cv2.line(mask,(cols-1,righty),(0,lefty),255,2)
#cv2.drawContours(mask,sorted_contours,0,255,-1)
pp = cv2.findNonZero(mask)

cv2.namedWindow("lab", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Min L", "lab", 0, 255, nothing)
cv2.createTrackbar("Min A", "lab", 0, 255, nothing)
cv2.createTrackbar("Min B", "lab", 0, 255, nothing)
cv2.createTrackbar("Max L", "lab", 0, 255, nothing)
cv2.createTrackbar("Max A", "lab", 0, 255, nothing)
cv2.createTrackbar("Max B", "lab", 0, 255, nothing)

lab_img = colour_img.copy()
lab_img = cv2.bitwise_and(colour_img, colour_img, mask=mask)
lab_img = cv2.cvtColor(lab_img, cv2.COLOR_BGR2LAB)
Z = lab_img.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 8 # clusters
ret, label, centre = cv2.kmeans(Z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centre = np.uint8(centre)
res = centre[label.flatten()]
res2 = res.reshape((colour_img.shape))
res_bgr = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
cv2.imshow("quantized", res_bgr)

for colour in centre:
    range_img = cv2.inRange(res2, colour, colour)
    cv2.imshow("LAB = [" + str(colour[0]) + ", " + str(colour[1]) + ", " + str(colour[2]) + "]", range_img)

cv2.waitKey(0)

"""
while True:
    lab_min = np.array((cv2.getTrackbarPos("Min L", "lab"), cv2.getTrackbarPos("Min A", "lab"), cv2.getTrackbarPos("Min B", "lab")))
    lab_max = np.array((cv2.getTrackbarPos("Max L", "lab"), cv2.getTrackbarPos("Max A", "lab"), cv2.getTrackbarPos("Max B", "lab")))
    range_img = cv2.inRange(res2, lab_min, lab_max)
    cv2.imshow("lab", range_img)

    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
"""
