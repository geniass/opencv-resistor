#!/usr/bin/python

import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow("original")
cv2.namedWindow('image')

img = cv2.imread("resistor.png", cv2.IMREAD_COLOR)

orig_gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = orig_gray_img.copy()
cv2.imshow("original", orig_gray_img)

cv2.createTrackbar("Min Thresh.", "image", 43, 512, nothing)
cv2.createTrackbar("Max Thresh.", "image",266, 512, nothing)

prev_d = 11
prev_sig_color = 10
prev_sig_space = 13
cv2.createTrackbar("d", "image", prev_d, 100, nothing)
cv2.createTrackbar("Sigma Colour", "image", prev_sig_color, 100, nothing)
cv2.createTrackbar("Sigma Space", "image", prev_sig_space, 100, nothing)

cv2.createTrackbar("Contour", "image", 0, 27, nothing)

while(1):

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    d = cv2.getTrackbarPos("d", "image")
    sig_color = cv2.getTrackbarPos("Sigma Colour", "image")
    sig_space = cv2.getTrackbarPos("Sigma Space", "image")
    if d != prev_d or sig_color != prev_sig_color or sig_space != prev_sig_space:
        gray_img = orig_gray_img.copy()
        gray_img = cv2.bilateralFilter(gray_img, d, sig_color, sig_space)
    min_thresh = cv2.getTrackbarPos("Min Thresh.", "image")
    max_thresh = cv2.getTrackbarPos("Max Thresh.", "image")
    edges = cv2.Canny(gray_img, min_thresh, max_thresh)
    cv2.imshow("image", edges)

#    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
#    img[dst > 0.01 * dst.max()] = [0,0,255]

    contrasted = cv2.equalizeHist(gray_img)
    thresh = cv2.adaptiveThreshold(contrasted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,9)

    cv2.imshow("original", thresh)

    """
    contour_img = img.copy()
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = cv2.getTrackbarPos("Contour", "image")
    if len(contours) >= i + 1:
        contour = contours[i]
        cv2.drawContours(contour_img, [contour], 0, (0, 255, 0), 1)

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        print len(approx)
        
        mask = np.zeros(gray_img.shape,np.uint8)
        cv2.drawContours(mask,[contour],0,255,-1)
        #pixelpoints = np.transpose(np.nonzero(mask))
        pixelpoints = cv2.findNonZero(mask)
        print cv2.mean(img, mask=mask)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    #cv2.drawContours(contour_img, contours, 0, (255, 0, 0), 1)
    cv2.imshow("original", contour_img)
    """

    prev_d = d
    prev_sig_color = sig_color
    prev_sig_space = sig_space

cv2.destroyAllWindows()
