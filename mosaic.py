import numpy as np
import cv2

def main():
    cap = cv2.VideoCapture('car_compressed.avi')
    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            i=i+1
            print i
            # cv2.imshow('frame',frame)
            name='frames/car/frame'+str(i)+'.png'
            # print name
            cv2.imwrite(name,frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print 'done'

if __name__ == '__main__':
    main()