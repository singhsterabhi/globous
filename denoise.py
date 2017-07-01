import cv2
# import numpy as np


def main():
    cap = cv2.VideoCapture('vid.MP4')
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('denoise', cv2.WINDOW_NORMAL)
    while(cap.isOpened()):
        ret, frame = cap.read()
        dst = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        cv2.imshow('frame', frame)
        cv2.imshow('denoise', dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    cv2.imread('img1.png')


if __name__ == '__main__':
    main()
