import numpy as np
# import bisect
# import cv2


def imadj(src, dst, tol=1, vmin=0, vmax=255):
    # src : input one-layer image (numpy array)
    # tol : tolerance, from 0 to 100.
    # vin  : src image bounds
    # vout : dst image bounds
    # return : output img
    print vmin, vmax
    # dst = src.copy()
    print dst.astype
    print dst.dtype
    # cv2.imshow('img1', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tol = max(0, min(100, tol))
    # print tol
    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.zeros(256, dtype=np.int)
        for r in range(src.shape[0]):
            for c in range(src.shape[1]):
                # print hist[src[r, c]]
                hist[src[r, c]] += 1
        # Cumulative histogram
        cum = hist.copy()
        for i in range(1, len(hist)):
            cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        # total = src.shape[0] * src.shape[1]
        # lobound = total * tol / 100
        # upp_bound = total * (100 - tol) / 100
        # print low_bound, upp_bound
        # vin[0] = bisect.bisect_left(cum, low_bound)
        # vin[1] = bisect.bisect_left(cum, upp_bound)
        # print vin
        # print vout

    # Stretching
    # scale = (vout[1] - vout[0]) / (vin[1] - vin[0] + 0.0001)
    scale = (vmax - vmin) / 255.0
    for r in range(dst.shape[0]):
        for c in range(dst.shape[1]):
            # print src[r, c]
            vs = int(scale * 255 * src[r, c]) - vmin
            vd = min(int(vs * scale + 0.5) + vmin, vmax)
            # print vs, vd
            dst[r, c] = int(vd)
            # print dst[r, c]
            # print int(vd)
    print dst
    # cv2.imshow('img', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst


# import cv2


# def main():
#     im = cv2.imread('img2.png')
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     # print gray
#     dst = gray.copy()
#     out = imadj(im, dst)
#     cv2.imshow('image', out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()
