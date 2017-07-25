# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:20:36 2017

@author: sagar iitm
"""

import cv2
import math
import numpy as np


def dark_channel(image, size):
    b, g, r = cv2.split(image)
    dark_channel_index = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_value = cv2.erode(dark_channel_index, kernel)
    return dark_channel_value


def atmospheric_light(image, dark_channel_value):
    [h, w] = image.shape[:2]
    image_size = h * w
    number_pixels = int(max(math.floor(image_size / 1000), 1))
    dark_channel_vector = dark_channel_value.reshape(image_size, 1)
    imvec = image.reshape(image_size, 3)

    indices = dark_channel_vector.argsort()
    indices = indices[image_size - number_pixels::]

    atm_sum = np.zeros([1, 3])
    for ind in range(1, number_pixels):
        atm_sum = atm_sum + imvec[indices[ind]]

    A = atm_sum / number_pixels
    return A


if __name__ == '__main__':
    import sys
    func = sys.argv[1]

    src = cv2.imread(func)

    I = src.astype('float64') / 255

    dark_channel_value = dark_channel(I, 15)
    A = atmospheric_light(I, dark_channel_value)
