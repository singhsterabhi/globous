import numpy as np
import cv2
import imadjust

# img, percen


def adjust():
    im = cv2.imread('img2.png')

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # print gray
    dst = gray.copy()
    print gray.astype
    print gray.dtype
    cv2.imshow('image', gray)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gray = im
    # if ~exist('percen','var') || isempty(percen), percen=[0.01 0.99]; end;
    # if not percen or 'percen' in locals():
    #     percen = [0.01, 0.99]
    # percen = [0.01, 0.99]

    # minn = min(min(gray[..., 0], gray[..., 1]), gray[..., 2])
    # gray = gray - minn
    # gray = gray / (max(max(gray[..., 0], gray[..., 1]), gray[..., 2]))

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    print minVal, maxVal

    gray = gray - minVal
    gray = gray / maxVal

    cv2.imshow('minmax', gray)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray1 = imadjust.imadj(gray, dst, vmin=minVal, vmax=maxVal)

    cv2.imshow('image1', gray1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def main():
#     adjust()


if __name__ == '__main__':
    adjust()
