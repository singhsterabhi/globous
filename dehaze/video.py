import cv2
import numpy as np
from non_local_dehazing import nonlocaldehazing
from atmlight_dcp import estairlight
# import im2double


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    # Divide all values by the largest possible value in the datatype
    return im.astype(np.float) / info.max


i = 0
# videoFReader = cv2.VideoCapture('vid.MP4')
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# v = cv2.VideoWriter('output.avi', fourcc, 12, (320, 240))
# # FrameRate = 12
# gamma = 1
# while(videoFReader.isOpened()):
#     print i
#     i = i + 1
#     if i == 30:
#         break
#     ret, image_hazy = videoFReader.read()
#     gamma = 1
#     if ret:
#         A = np.reshape(estairlight(np.power(im2double(image_hazy), gamma)), (1, 1, 3))
#         (image_dehazed, transmission_refined) = nonlocaldehazing(image_hazy, A, gamma)
#         v.write(image_dehazed)

# v.release()
# videoFReader.release()

# image
gamma = 1
image_hazy = cv2.imread('pumpkins.png')
A = np.reshape(estairlight(np.power(im2double(image_hazy), gamma)), (1, 1, 3))
(image_dehazed, transmission_refined) = nonlocaldehazing(image_hazy, A, gamma)

cv2.imshow('imagedehazed', image_dehazed)
cv2.waitKey(0)
cv2.destroyAllWindows()
