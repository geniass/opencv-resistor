#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


def find_lines(img, edges):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines3.jpg', img)

global hsv_img

img = cv2.imread("blue.jpg")
img = cv2.copyMakeBorder(
    img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
colour_img = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

th_a = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
cv2.imshow("adaptive", th_a)
th3 = th_a

cv2.imshow("blur", blur)
cv2.imshow("otsu", th3)

# erosion = cv2.erode(blur, np.ones((29, 1), np.uint8))
# cv2.imshow("erode", erosion)

edges = cv2.Canny(th3, 20, 200)
cv2.imshow("edges", edges)

contour_img = img.copy()
_, contours, hier = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(cv2.contourArea(sorted_contours[0]))

cv2.drawContours(contour_img, sorted_contours, 0, (0, 255, 0), 3)
cv2.imshow("contours", contour_img)

peri = cv2.arcLength(sorted_contours[0], True)
approx = cv2.approxPolyDP(sorted_contours[0], 0.02 * peri, True)

mask = np.zeros(img.shape, np.uint8)
rows, cols = img.shape[:2]
#[vx, vy, x, y] = cv2.fitLine(
#    sorted_contours[0], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
#lefty = int((-x * vy / vx) + y) + 15
#righty = int(((cols - x) * vy / vx) + y) + 15
#cv2.line(mask, (cols - 1, righty), (0, lefty), 255, 2)
# cv2.drawContours(mask,sorted_contours,0,255,-1)
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
Z = lab_img.reshape((-1, 3))
Z = np.float32(Z)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# k = 8  # clusters
# ret, label, centre = cv2.kmeans(Z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# centre = np.uint8(centre)
# res = centre[label.flatten()]
# res2 = res.reshape((colour_img.shape))
# res_bgr = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)
# cv2.imshow("quantized", res_bgr)

# for colour in centre:
#     range_img = cv2.inRange(res2, colour, colour)
#     cv2.imshow("LAB = [" + str(colour[0]) + ", " +
#                str(colour[1]) + ", " + str(colour[2]) + "]", range_img)


rect = cv2.minAreaRect(sorted_contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
print(box[1][1], box[0][1])
print(box[0][0], box[2][0])
# TODO: dont hard code this shit
roi_img = img[box[1][1]:box[0][1], box[2][0]:box[0][0]]
cv2.drawContours(img, [box], 0, (128, 255, 0), 2)
cv2.imshow("Image", img)

th_a = cv2.adaptiveThreshold(
    roi_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
cv2.imshow("roi", th_a)


# roi_img = cv2.GaussianBlur(roi_img, (5, 15), 0)
gradx = cv2.Sobel(roi_img, cv2.CV_64F, 1, 0, ksize=-1)
grady = cv2.Sobel(roi_img, cv2.CV_64F, 0, 1, ksize=-1)
grad = cv2.subtract(gradx, grady)
# grad = cv2.Laplacian(roi_img, cv2.CV_64F)
cv2.imshow("grad", grad)


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
