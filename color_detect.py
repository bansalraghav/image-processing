#!/usr/bin/env python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#function to detect green color. It returns the frame after drawing the contour on it.
def greenCircleDetect(frame):

	# converting from BGR to HSV color space
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# HSV Range for green
	lower_green = np.array([36,25,25])
	upper_green = np.array([75,255,255])
	mask = cv2.inRange(hsv, lower_green, upper_green) # generating a mask in the image to determine the region of the detected color
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 2) #morphological operation done on the image to remove the false positives.

	_, contours, _  = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #extracting the contours from the image
	output  = cv2.drawContours(frame, contours, -1, (255,0,0),3) #drawing all possible detected contours with blue color.
	return output

#function to detect red color. It returns the frame after drawing the contour on it.
def redCircleDetect(frame):

	# converting from BGR to HSV color space
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# HSV Range for lower red
	lower_red = np.array([0,230,20])
	upper_red = np.array([1,255,255])
	mask1 = cv2.inRange(hsv, lower_red, upper_red)

	# HSV Range for upper red
	lower_red = np.array([170,120,20])
	upper_red = np.array([180,255,255])
	mask2 = cv2.inRange(hsv,lower_red,upper_red)

	# Generating the final mask to detect red color
	mask1 = mask1+mask2
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask1,kernel,iterations = 2) #morphological operation done on the image to remove the false positives.

	_, contours, _  = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #extracting all possible contours from the image
	output  = cv2.drawContours(frame, contours, -1, (255,0,0),3) #drawing all possible detected contours with blue color.
	return output


#function to detect blue color. It returns the frame after drawing the contour on it.
def blueCircleDetect(frame):

	# converting from BGR to HSV color space
	hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	# HSV Range for blue
	lower_blue = np.array([85,86,0])
	upper_blue = np.array([130,255,255])
	mask = cv2.inRange(hsv, lower_blue, upper_blue) # generating a mask in the image to determine the region of the detected color
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(mask,kernel,iterations = 2) #morphological operation done on the image to remove the false positives.

	_, contours, _  = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #extracting all possible contours from the image
	output  = cv2.drawContours(frame, contours, -1, (255,0,0),3) #drawing all possible detected contours with blue color.
	return output

while True:
	ret, frame = cap.read()

	green = greenCircleDetect(frame)
	red = redCircleDetect(frame)
	blue = blueCircleDetect(frame)

	cv2.imshow('frame', frame)
	cv2.waitKey(3)

cap.release()
cv2.destroyAllWindows()
