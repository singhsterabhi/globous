import cv2
import sys


# print len(sys.argv)
# sys.stdout.flush()
# print str(sys.argv)
# sys.stdout.flush()

video = str(sys.argv[1])
clip = int(sys.argv[2])

videoFReader = cv2.VideoCapture(video)
fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
v = cv2.VideoWriter('/mnt/01D2BD9F53868420/x/clahe/output.avi', fourcc, 12, (3840, 2160))
# 3840 x 2160
# FrameRate = 12
# gamma = 1
i = 0
length = int(videoFReader.get(cv2.CAP_PROP_FRAME_COUNT))
# print length
sys.stdout.flush()
for i in range(0, length):
    # print i
    # i = i + 1
    # if i == 150:
    #     break
    ret, img = videoFReader.read()
    gamma = 1
    if ret:
        print i
        sys.stdout.flush()
        # -----Converting image to LAB Color model-----------------------------------
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # cv2.imshow("lab", lab)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        # cv2.imshow('l_channel', l)
        # cv2.imshow('a_channel', a)
        # cv2.imshow('b_channel', b)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # cv2.imshow('CLAHE output', cl)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))
        # cv2.imshow('limg', limg)

        # -----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # cv2.imshow('final', final)

        v.write(final)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
v.release()
videoFReader.release()
