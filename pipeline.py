import numpy as np

import cbir


def pipe(img):
    C = cbir.preprocess(img)
    W = cbir.wavelet_transform(C, 3)
    W_ = cbir.wavelet_transform(C, 4)
    return cbir.generate_feature(W, W_)


def filter_(f, f_):
    if cbir.match(f[2], f_[2]):
        dist = cbir.dist(f[1], f_[1])
        return dist
    return np.Inf
