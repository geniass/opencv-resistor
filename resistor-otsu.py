#!/usr/bin/python

import cv2
import numpy as np

def nothing(x):
    pass

global hsv_img

img = cv2.imread("resistor.png")
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
sorted_contours = sorted([cv2.contourArea(contour) for contour in contours])

cv2.drawContours(contour_img,contours, 4, (0,255,0), 3)
cv2.imshow("contours", contour_img)


mask = np.zeros(img.shape,np.uint8)
rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(contours[4], cv2.cv.CV_DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(mask,(cols-1,righty),(0,lefty),255,2)
pp = cv2.findNonZero(mask)

img = cv2.bitwise_and(blur, mask)
#ret,thr = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,9)
#img = cv2.equalizeHist(img)
cv2.imshow("line", img)

hsv_img = colour_img.copy()
print hsv_img.size
print mask.size
hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("hsv", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Min H", "hsv", 0, 255, nothing)
cv2.createTrackbar("Min S", "hsv", 0, 255, nothing)
cv2.createTrackbar("Min V", "hsv", 0, 255, nothing)
cv2.createTrackbar("Max H", "hsv", 0, 255, nothing)
cv2.createTrackbar("Max S", "hsv", 0, 255, nothing)
cv2.createTrackbar("Max V", "hsv", 0, 255, nothing)


while True:
    hsv_min = np.array((cv2.getTrackbarPos("Min H", "hsv"), cv2.getTrackbarPos("Min S", "hsv"), cv2.getTrackbarPos("Min V", "hsv")))
    hsv_max = np.array((cv2.getTrackbarPos("Max H", "hsv"), cv2.getTrackbarPos("Max S", "hsv"), cv2.getTrackbarPos("Max V", "hsv")))
    #hsv_min = np.array([0,0,0])
    #hsv_max = np.array([0,0,0])

    range_img = cv2.inRange(hsv_img, hsv_min, hsv_max)

    cv2.imshow("hsv", range_img)

    k = cv2.waitKey(5)
    if k == 27:
        break
cv2.destroyAllWindows()
