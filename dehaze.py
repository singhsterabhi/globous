# import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture('vid.MP4')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('denoise', cv2.WINDOW_NORMAL)
    # i = 0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (3840, 2160))

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (3840,2160))
    # ret, frame = cap.read()
    while(cap.isOpened()):
        ret, frame = cap.read()
        # i=i+1
        # print i
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,11)
            # dst = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
            cv2.imshow('frame', frame)
            cv2.imshow('gray', gray)
            # cv2.imshow('thresh',thresh1)
            # cv2.imshow('denoise',dst)
            # cv2.imwrite('img.png',frame)
            # out.write(dst)
            # if i==200:
            #     break
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if i==20:
            #     cv2.imwrite('img1.png',frame)
            # if i==35:
            #     cv2.imwrite('img2.png',frame)
            #     break
            # break
            # k = cv2.waitKey(0)
            # if k == 27:         # wait for ESC key to exit
            #     cv2.destroyAllWindows()
            # elif k == ord('s'): # wait for 's' key to save and exit
            #     cv2.imwrite('messigray.png',img)
            #     cv2.destroyAllWindows()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print 'done'


if __name__ == '__main__':
    main()
