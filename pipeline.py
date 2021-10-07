import numpy as np

import cbir


class Pipeline:
    def __init__(self, threshold):
        self.threshold = threshold

    def pipe(self, img):
        C = cbir.preprocess(img)
        W = cbir.wavelet_transform(C, 3)
        W_ = cbir.wavelet_transform(C, 4)
        return cbir.generate_feature(W, W_)

    def filter_(self, f, f_):
        if cbir.match(f[0], f_[0]):
            dist = cbir.dist(f[1], f_[1])
            if dist < self.threshold:
                return cbir.dist(f[2], f_[2])
        return np.Inf
