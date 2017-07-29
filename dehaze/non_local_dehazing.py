#!python numbers=disable

import numpy as np
# import cv2
# import math

# accumarray
from itertools import product

# from adjust import adjust

from wlsoptim import wls

from scipy import spatial


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image
    # Divide all values by the largest possible value in the datatype
    return im.astype(np.float) / info.max


def nonlocaldehazing(img_hazy, air_light, gamma):

        # validating inputs
    h, w, n_colors = img_hazy.shape
    if n_colors != 3:
        print 'error. rgb image required'
        return
    # print hash(tuple(air_light))
    # if not (air_light in locals()) or (not air_light) or ((air_light.size) != 3):
    #     print 'air light required'
    #     return
    if not (gamma in locals()) or (not gamma):
        gamma = 1

    img_hazy = im2double(img_hazy)
    # radiometric correction
    img_hazy_corrected = np.power(img_hazy, gamma)

    # find Haze-Lines
    # Translate the coordinate system to be air_light-centric (Eq. (3)
    dist_from_airlight = np.zeros((h, w, n_colors))
    for color_idx in range(0, n_colors):
        dist_from_airlight[:, :, color_idx] = img_hazy_corrected[:, :, color_idx] - air_light[:, :, color_idx]

    # Calculate radius (Eq. (5))
    radius = np.sqrt(((dist_from_airlight[:, :, 0])**2) + ((dist_from_airlight[:, :, 1])**2) + ((dist_from_airlight[:, :, 2])**2))

    # Cluster the pixels to haze-lines
    # Use a KD-tree impementation for fast clustering according to their angles
    dist_unit_radius = np.reshape(dist_from_airlight, (h * w, n_colors))
    # possibility of error ################
    dist_norm = np.sqrt(np.sum((dist_unit_radius**2), axis=0))

    # bsxfun rdivide
    # error of NAN can occur ##############
    dist_unit_radius = dist_norm / dist_unit_radius

    n_points = 1000

    # load pre-calculated uniform tesselation of the unit-sphere
    # fid = open('TR1000.txt', 'r')

    # ### cell2mat
    points = np.loadtxt('tr1000.csv', delimiter=',')

    # mdl = kdtree(points)
    # print dist_unit_radius
    # print points
    ind = knn_search(points, 1)
    print ind
    # Estimating Initial Transmission
    # Estimate radius as the maximal radius in each haze-line (Eq. (11))
    # print radius
    # K = accumarray
    K = accum(ind, radius[:], size=[n_points, 1], func=lambda x: max(x))

    radius_new = np.reshape(K(ind), (h, w))

    # Estimate transmission as radii ratio (Eq. (12))
    transmission_estimation = radius / radius_new

    # Limit the transmission to the range [trans_min, 1] for numerical stability
    trans_min = 0.1
    transmission_estimation = np.minimim(np.maximum(transmission_estimation, trans_min), 1)

    # Regularization
    # Apply lower bound from the image (Eqs. (13-14))
    trans_lower_bound = 1 - np.minimum((np.reshape(air_light, (1, 1, 3))) / img_hazy_corrected, [], 3)
    transmission_estimation = np.maximum(transmission_estimation, trans_lower_bound)

    # Solve optimization problem (Eq. (15))
    # find bin counts for reliability - small bins (#pixels<50)
    # do not comply with
    # the model assumptions and should be disregarded
    bin_count = accum(ind, 1, size=[n_points, 1])
    bin_count_map = np.reshape(bin_count(ind), (h, w))
    # bin_eval_fun = binevalfun(x)

    # Calculate std - this is the data-term weight of Eq. (15)
    K_std = accum(ind, radius[:], size=[n_points], func=lambda x: np.std(x))
    radius_std = np.reshape(K_std(ind), (h, w))
    # radius_eval_fun = radevalfun(x)
    radius_reliability = radevalfun(radius_std / radius_std.max(0))
    data_term_weight = binevalfun(bin_count_map) * radius_reliability
    lamda = 0.1
    transmission = wls(transmission_estimation, data_term_weight, img_hazy)

    # Dehazing
    #  eq 16
    img_dehazed = np.zeros((h, w, n_colors))
    leave_haze = 1.06
    for color_idx in range(0, 3):
        img_dehazed[:, :, color_idx] = (img_hazy_corrected[:, :, color_idx] - (1 - leave_haze * transmission) * air_light(color_idx)) / np.maximum(transmission, trans_min)

    # Limit each pixel value to the range [0, 1] (avoid numerical problems)
    img_dehazed[img_dehazed > 1] = 1
    img_dehazed[img_dehazed < 0] = 0
    img_dehazed = img_dehazed**(1 / gamma)

    # For display, we perform a global linear contrast stretch on the output,
    # clipping 0.5% of the pixel values both in the shadows and in the highlights
    # adj_percent = [0.005, 0.995]
    # img_dehazed = adjust(img_dehazed, adj_percent)
    img_dehazed.astype('uint8')

    return (img_dehazed, transmission)


def radevalfun(r):
    return np.minimum(1, 3 * np.maximum(0.001, r - 0.1))


def binevalfun(x):
    return np.minimum(1, x / 50)


# import np

def kdtree(data, leafsize=10):
    """
    build a kd-tree for O(n log n) nearest neighbour search

    input:
        data:       2D ndarray, shape =(ndim,ndata), preferentially C order
        leafsize:   max. number of data points to leave in a leaf

    output:
        kd-tree:    list of tuples
    # """

    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = np.zeros((2, data.shape[0]))
    hrect[0, :] = data.min(axis=1)
    hrect[1, :] = data.max(axis=1)

    # create root of kd-tree
    idx = np.argsort(data[0, :], kind='mergesort')
    data[:, :] = data[:, idx]
    splitval = data[0, ndata / 2]

    left_hrect = hrect.copy()
    right_hrect = hrect.copy()
    left_hrect[1, 0] = splitval
    right_hrect[0, 0] = splitval

    tree = [(None, None, left_hrect, right_hrect, None, None)]

    stack = [(data[:, :ndata / 2], idx[:ndata / 2], 1, 0, True),
             (data[:, ndata / 2:], idx[ndata / 2:], 1, 0, False)]

    # recursively split data in halves using hyper-rectangles:
    while stack:

        # pop data off stack
        data, didx, depth, parent, leftbranch = stack.pop()
        ndata = data.shape[1]
        nodeptr = len(tree)

        # update parent node

        _didx, _data, _left_hrect, _right_hrect, left, right = tree[parent]

        tree[parent] = (_didx, _data, _left_hrect, _right_hrect, nodeptr, right) if leftbranch \
            else (_didx, _data, _left_hrect, _right_hrect, left, nodeptr)

        # insert node in kd-tree

        # leaf node?
        if ndata <= leafsize:
            _didx = didx.copy()
            _data = data.copy()
            leaf = (_didx, _data, None, None, 0, 0)
            tree.append(leaf)

        # not a leaf, split the data in two
        else:
            splitdim = depth % ndim
            idx = np.argsort(data[splitdim, :], kind='mergesort')
            data[:, :] = data[:, idx]
            didx = didx[idx]
            nodeptr = len(tree)
            stack.append((data[:, :ndata / 2], didx[:ndata / 2], depth + 1, nodeptr, True))
            stack.append((data[:, ndata / 2:], didx[ndata / 2:], depth + 1, nodeptr, False))
            splitval = data[splitdim, ndata / 2]
            if leftbranch:
                left_hrect = _left_hrect.copy()
                right_hrect = _left_hrect.copy()
            else:
                left_hrect = _right_hrect.copy()
                right_hrect = _right_hrect.copy()
            left_hrect[1, splitdim] = splitval
            right_hrect[0, splitdim] = splitval
            # append node to tree
            tree.append((None, None, left_hrect, right_hrect, None, None))

    return tree


def knn_search(data, K, leafsize=2048):
    """ find the K nearest neighbours for data points in data,
        using an O(n log n) kd-tree """

    ndata = data.shape[1]
    param = data.shape[0]

    # build kdtree
    tree = kdtree(data.copy(), leafsize=leafsize)

    # search kdtree
    knn = []
    for i in np.arange(ndata):
        _data = data[:, i].reshape((param, 1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K + 1)
        knn.append(_knn[1:])

    return knn


def search_kdtree(tree, datapoint, K):
    """ find the k nearest neighbours of datapoint in a kdtree """
    stack = [tree[0]]
    knn = [(np.inf, None)] * K
    _datapt = datapoint[:, 0]
    while stack:

        leaf_idx, leaf_data, left_hrect, right_hrect, left, right = stack.pop()

        # leaf
        if leaf_idx is not None:
            _knn = quadratic_knn_search(datapoint, leaf_idx, leaf_data, K)
            if _knn[0][0] < knn[-1][0]:
                knn = sorted(knn + _knn)[:K]

        # not a leaf
        else:

            # check left branch
            if intersect(left_hrect, knn[-1][0], _datapt):
                stack.append(tree[left])

            # chech right branch
            if intersect(right_hrect, knn[-1][0], _datapt):
                stack.append(tree[right])
    return knn


def intersect(hrect, r2, centroid):
    """
    checks if the hyperrectangle hrect intersects with the
    hypersphere defined by centroid and r2
    """
    maxval = hrect[1, :]
    minval = hrect[0, :]
    p = centroid.copy()
    idx = p < minval
    p[idx] = minval[idx]
    idx = p > maxval
    p[idx] = maxval[idx]
    return ((p - centroid)**2).sum() < r2


def quadratic_knn_search(data, lidx, ldata, K):
    """ find K nearest neighbours of data among ldata """
    ndata = ldata.shape[1]
    param = ldata.shape[0]
    K = K if K < ndata else ndata
    retval = []
    sqd = ((ldata - data[:, :ndata])**2).sum(axis=0)  # data.reshape((param,1)).repeat(ndata, axis=1);
    idx = np.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    accmap = np.asarray(accmap)
    a = np.asarray(a)
    print accmap.shape[:a.ndim]
    print a.shape
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out
