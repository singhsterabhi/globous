import cv2

im=cv2.imread('frame79.jpg')
h=im.shape[0]
w=im.shape[1]
print h
print w
im=im[:,(30):(w-30)]
# cv2.namedWindow('window')
cv2.imwrite('half3.jpg',im)
# cv2.imshow('window',im)
# cv2.waitKey(0)
cv2.destroyAllWindows()