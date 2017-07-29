#!python numbers=disable

# Copyleft 2008 Sturla Molden
# University of Oslo

#import psyco
# psyco.full()

import numpy


def kdtree(data, leafsize=10):
    """
    build a kd-tree for O(n log n) nearest neighbour search

    input:
        data:       2D ndarray, shape =(ndim,ndata), preferentially C order
        leafsize:   max. number of data points to leave in a leaf

    output:
        kd-tree:    list of tuples
    """

    ndim = data.shape[0]
    ndata = data.shape[1]

    # find bounding hyper-rectangle
    hrect = numpy.zeros((2, data.shape[0]))
    hrect[0, :] = data.min(axis=1)
    hrect[1, :] = data.max(axis=1)

    # create root of kd-tree
    idx = numpy.argsort(data[0, :], kind='mergesort')
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
            idx = numpy.argsort(data[splitdim, :], kind='mergesort')
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


#!python numbers=disable


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
    idx = numpy.argsort(sqd, kind='mergesort')
    idx = idx[:K]
    return zip(sqd[idx], lidx[idx])


def search_kdtree(tree, datapoint, K):
    """ find the k nearest neighbours of datapoint in a kdtree """
    stack = [tree[0]]
    knn = [(numpy.inf, None)] * K
    _datapt = datapoint[:, 0]
    while stack:

        leaf_idx, leaf_data, left_hrect, \
            right_hrect, left, right = stack.pop()

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


def radius_search(tree, datapoint, radius):
    """ find all points within radius of datapoint """
    stack = [tree[0]]
    inside = []
    while stack:

        leaf_idx, leaf_data, left_hrect, \
            right_hrect, left, right = stack.pop()

        # leaf
        if leaf_idx is not None:
            param = leaf_data.shape[0]
            distance = numpy.sqrt(((leaf_data - datapoint.reshape((param, 1)))**2).sum(axis=0))
            near = numpy.where(distance <= radius)
            if len(near[0]):
                idx = leaf_idx[near]
                distance = distance[near]
                inside += (zip(distance, idx))

        else:

            if intersect(left_hrect, radius, datapoint):
                stack.append(tree[left])

            if intersect(right_hrect, radius, datapoint):
                stack.append(tree[right])

    return inside

if __name__ == '__main__':
    points = numpy.loadtxt('tr1000.csv', delimiter=',')
    ind = knn_search(points, 1)
    print ind
