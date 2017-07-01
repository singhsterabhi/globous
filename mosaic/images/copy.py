import cv2
import numpy as np

im=cv2.imread('im.jpg')
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
print im.shape
im1=im[0:1200,0:450]
im2=im[0:1200,400:950]
im3=im[0:1200,850:1350]
im4=im[0:1200,1000:1600]
im5=im[0:1200,1500:1920]
# cv2.imshow('big hero 62',im1)
cv2.imwrite('s1.jpg',im1)
cv2.imwrite('s2.jpg',im2)
cv2.imwrite('s3.jpg',im3)
cv2.imwrite('s4.jpg',im4)
cv2.imwrite('s5.jpg',im5)
cv2.waitKey(0)
cv2.destroyAllWindows()