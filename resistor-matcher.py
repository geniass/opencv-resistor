#!/usr/bin/python

import cv2
from matplotlib import pyplot as plt
#from find_obj import filter_matches,explore_match

scene_img = cv2.imread("resistor.png", cv2.IMREAD_GRAYSCALE)
template_img = cv2.imread("band.png", cv2.IMREAD_GRAYSCALE)
#scene_img = cv2.fastNlMeansDenoising(scene_img, None, 10, 7, 21)
#scene_img = cv2.equalizeHist(scene_img)

#detector = cv2.ORB()
detector = cv2.SURF(85)

scene_keypoints, scene_desc = detector.detectAndCompute(scene_img, None)
template_keypoints, template_desc = detector.detectAndCompute(template_img, None)


print len(scene_keypoints)
print len(template_keypoints)

img3 = cv2.drawKeypoints(template_img, template_keypoints)
cv2.imshow("Template Features", img3)
img2 = cv2.drawKeypoints(scene_img, scene_keypoints)
cv2.imshow("Scene Features", img2)

"""
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(scene_desc, template_desc)
matches = sorted(matches, key = lambda x:x.distance)
"""

#p1, p2, kp_pairs = filter_matches(scene_keypoints, template_keypoints, matches)
#explore_match('find_obj', scene_img, template_img,kp_pairs)#cv2 shows image
#img = cv2.drawMatches(scene_img, scene_keypoints, template_img, template_keypoints, matches[:10], flags=2)
#plt.imshow(img),plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
