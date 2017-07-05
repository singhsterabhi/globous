# http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png 

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2



# # cap = cv2.VideoCapture('car_compressed.avi')
# # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
# i=0

# result = cv2.imread('frames/frame1.jpg')

# stitcher = Stitcher()

# for j in range(2,385):
# 	# ret, frame = cap.read()
# 	print j
# 	imageA=result
# 	name='frames/frame'+str(j)+'.jpg'
# 	imageB=cv2.imread(name)
# 	(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
# 	name='result.jpg'
# 	cv2.imwrite(name,result)
	
# print 'done'



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
# imageA = imutils.resize(imageA, width=400)
# imageB = imutils.resize(imageB, width=400)

# stitch the images together to create a panorama
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imwrite('result1.png',result)
cv2.imwrite('keymat.jpg',vis)

cv2.namedWindow('Keypoint Matches')
cv2.namedWindow('Result')

cv2.namedWindow('Image A')
cv2.namedWindow('Image B')

cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)