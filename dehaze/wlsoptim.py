# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:19:27 2017

@author: sagar iitm
"""
# import cv2
import math
import numpy as np
# import sys
import scipy
from sklearn import preprocessing


def wls(inp, data_weight, guidance):
    small_num = 0.0001
    lambdav = 0.05
    # image = sys.argv[1]
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # guidance = np.log(gray_image)
    [h, w] = guidance.shape[:2]
    k = h * w

    dy1 = np.diff(guidance)
    dy1 = np.divide(-lambdav, (np.sum(np.square(math.fabs(dy1)), 2) + small_num))
    npad = ((0, 0), (1, 2), (2, 1))
    dy1 = np.pad(dy1, pad_width=npad, mode='constant', constant_values=0)
    shapedy1 = np.shape(dy1)
    dy = np.reshape(dy1, (shapedy1[0] * shapedy1[1], 1))

    dx1 = np.diff(guidance)
    dx1 = np.divide(-lambdav, (np.sum(np.square(math.fabs(dx1)), 2) + small_num))
    npad = ((0, 0), (1, 2), (2, 1))
    dx1 = np.pad(dx1, pad_width=npad, mode='constant', constant_values=0)
    shapedx1 = np.shape(dx1)
    dx = np.reshape(dx1, (shapedx1[0] * shapedx1[1], 1))

    shapedy = np.shape(dy)
    shapedx = np.shape(dx)

    # Construct a five-point spatially inhomogeneous Laplacian matrix
    B = np.concatenate((dx, dy), axis=1)
    d = [-h, -1]
    tmp = scipy.sparse.spdiags(B, d, k, k, format=None)

    ea = dx
    we = np.zeroe(h + shapedx[0], 1)
    # we = padarray(dx, h, 'pre');
    we = we[0:len(we) - h + 1]
    so = dy
    no = np.zeroe(1 + shapedy[0], 1)
    # no = padarray(dy, 1, 'pre');
    no = no[0:len(no) - 2]

    D = -(ea + we + so + no)
    spdiags1a = scipy.sparse.spdiags(D, 0, k, k, format=None)
    tmpt = np.transpose(tmp)
    Asmoothness = tmp + tmpt + spdiags1a

    # Normalize data weight
    # data_weight = data_weight - np.min(np.min(np.min(data_weight, axis=1), axis=0))
    # data_weight = 1.*data_weight./(max(data_weight(:))+small_num);
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data_weight)
    data_weight = X_train_minmax

    # Make sure we have a boundary condition for the top line:
    # It will be the minimum of the transmission in each column
    # With reliability 0.8
    reliability_mask = (data_weight[1, :] < 0.6)  # find missing boundary condition
    in_row1 = inp.min(axis=0)
    data_weight[1, reliability_mask] = 0.8
    inp[1, reliability_mask] = in_row1[reliability_mask]

    shapedw = np.shape(data_weight)
    dw = np.reshape(data_weight, (shapedw[0] * shapedw[1], 1))

    Adata = scipy.sparse.spdiags(dw, 0, k, k)

    A = Adata + Asmoothness

    shapeinp = np.shape(inp)
    inpv = np.reshape(inp, (shapeinp[0] * shapeinp[1], 1))

    b = Adata * inpv

    # Solve
    # out = lsqnonneg(A,b);
    out = A / b
    out = np.reshape(out, (h, w))
    return out
