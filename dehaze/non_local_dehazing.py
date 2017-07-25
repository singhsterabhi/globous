import numpy as np
import cv2
import math


def non_local_dehazing(img_hazy, air_light, gamma):

        # validating inputs
    h, w, n_colors = img_hazy.shape
    if n_colors != 3:
        print 'error. rgb image required'
        return
    if not (air_light in locals()) or (not air_light) or (size(air_light) != 3):
        print 'air light required'
        return
    if not (gamma in locals()) or (not gamma):
        gamma = 1

    img_hazy = im2double(img_hazy)
    # radiometric correction
    img_hazy_corrected = np.power(img_hazy, gamma)

    # find Haze-Lines
    # Translate the coordinate system to be air_light-centric (Eq. (3)
    dist_from_airlight = np.zeros((h, w, n_colors))
    for color_idx in range(0, n_colors):
        dist_from_airlight(: , : , color_idx) = img_hazy_corrected(: , : , color_idx)-air_light(: , : , color_idx)

    # Calculate radius (Eq. (5))
    radius = math.sqrt(math.pow(dist_from_airlight(:, : , 1), 2)+math.pow(dist_from_airlight(: , : , 2), 2)+math.pow(dist_from_airlight(: , : , 3), 2))

    # Cluster the pixels to haze-lines

    # Use a KD-tree impementation for fast clustering according to their angles
    dist_unit_radius = np.reshape(dist_from_airlight, (h * w, n_colors))
    # possibility of error ################
    dist_norm = math.sqrt(np.sum(np.power(dist_unit_radius, 2), axis=0))

    # bsxfun rdivide
    # error of NAN can occur ##############
    dist_unit_radius = dist_norm / dist_unit_radius

    n_points = 1000

    # load pre-calculated uniform tesselation of the unit-sphere
    fid = open('TR1000.txt', 'r')
    points = np.loadtxt(fid, delimiter=' ')

    # ### cell2mat

    # mdl = kdtree(points)

    ind = knn_search(points, dist_unit_radius)
    
    # Estimating Initial Transmission
    # Estimate radius as the maximal radius in each haze-line (Eq. (11))

    # K = accumarray

    radius_new = np.reshape(K(ind), (h, w))

    # Estimate transmission as radii ratio (Eq. (12))
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
    bin_count = accumarray(ind, radius(: ), [n_points, ], @std)
    bin_count_map = np.reshape(bin_count(ind), (h, w))
    # bin_eval_fun = binevalfun(x)

    # Calculate std - this is the data-term weight of Eq. (15)
    K_std = accumarray(ind, radius(: ), [n_points], '@std')
    radius_std = np.reshape(K_std(ind), (h, w))
    # radius_eval_fun = radevalfun(x)
    radius_reliability = radevalfun(radius_std / radius_std.max(0))
    data_term_weight = binevalfun(bin_count_map) * radius_reliability
    lamda = 0.1
    transmission = wls_optimization(transmission_estimation, data_term_weight, img_hazy, lamda)

    # Dehazing
    #  eq 16
    img_dehazed = np.zeros((h, w, n_colors))
    leave_haze = 1.06
    for color_idx in range(0, 3):
        img_dehazed(:, : , color_idx) = (img_hazy_corrected(: , : , color_idx) - (1-leave_haze*transmission)*air_light(color_idx))/np.maximum(transmission, trans_min)

    # Limit each pixel value to the range [0, 1] (avoid numerical problems)
    img_dehazed[img_dehazed > 1] = 1
    img_dehazed[img_dehazed < 0] = 0
    img_dehazed = img_dehazed**(1 / gamma)

    # For display, we perform a global linear contrast stretch on the output,
    # clipping 0.5% of the pixel values both in the shadows and in the highlights
    adj_percent = [0.005, 0.995]
    img_dehazed = adjust(img_dehazed, adj_percent)
    img_dehazed.astype('uint8')


def radevalfun(r):
    return np.minimum(1, 3 * np.maximum(0.001, r - 0.1))


def binevalfun(x):
    return np.minimum(1, x / 50)


def accumarray(sub, val, sz, fun='@sum'):
    pass

    # sales.rep    = [1 2 1 3 4 6 2 3 1 1];
    # sales.amount = [1 1 3 1 5 2 2 4 1 5];

    # numReps = max(sales.rep);
    # sales.total   = zeros(1,numReps);
    # for i = 1 : numel(sales.rep)
    #     sales.total(sales.rep(i)) = sales.total(sales.rep(i)) ...
    #                               + sales.amount(i);
    # end

    # sales.accum = accumarray(sales.rep', sales.amount)'


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
    for i in numpy.arange(ndata):
        _data = data[:, i].reshape((param, 1)).repeat(leafsize, axis=1)
        _knn = search_kdtree(tree, _data, K + 1)
        knn.append(_knn[1:])

    return knn
