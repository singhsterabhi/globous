import numpy as np
from accum import accum


def main():
    a = np.array([[1, 2, 3], [4, -1, 6], [-1, 8, 9]])
    accmap = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    month = np.array([1, 1, 2, 3, 8, 1, 3, 4, 9, 11, 9, 12, 3, 4, 11])
    temp = np.array([57, 61, 60, 62, 45, 59, 64, 66, 40, 56, 38, 65, 61, 64, 55])
    result = accum(month, temp, func=lambda x: np.std(x))
    print result

if __name__ == '__main__':
    main()
