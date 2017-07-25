import cv2
import numpy as np
import non_local_dehazing
import estimate_airlight
import im2double


i = 0
videoFReader = cv2.VideoCapture('vid.MP4')
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
v = cv2.VideoWriter('output.avi', fourcc, 12, (320, 240))
# FrameRate = 12
gamma = 1
while(videoFReader.isOpened()):
    i = i + 1
    ret, image_hazy = videoFReader.read()
    gamma = 1
    if ret:
        A = np.reshape(estimate_airlight(
            np.power(im2double(image_hazy), gamma)), [1, 1], 3)
        (image_dehazed, transmission_refined) = non_local_dehazing(
            image_hazy, A, gamma)
        v.write(image_dehazed)

v.release()
videoFReader.release()
