import numpy as np

import cbir


class Pipeline:
    def __init__(self, threshold=30000, vertical=False, horizontal=False, diagonal=False, color=False, scale=1.5):
        self.threshold = threshold
        self.vertical = vertical
        self.horizontal = horizontal
        self.diagonal = diagonal
        self.color = color
        self.w_jk = np.array([[1, 1], [1, 1]], dtype=np.float32)
        self.w_c = np.array([1, 1, 1], dtype=np.float32)
        self.decomposition_levels = np.array([3, 4], dtype=np.float32)
        self.adjust_weights(scale)

    def process(self, img):
        C = cbir.preprocess(img)
        W = cbir.wavelet_transform(C, self.decomposition_levels[0])
        W_ = cbir.wavelet_transform(C, self.decomposition_levels[1])
        return cbir.generate_feature(W, W_)

    def filter_(self, f, f_):
        if cbir.match(f[0], f_[0]):
            dist = cbir.dist(f[1], f_[1], weights_jk=self.w_jk, weights_c=self.w_c)
            if dist < self.threshold:
                return cbir.dist(f[2], f_[2], weights_jk=self.w_jk, weights_c=self.w_c)
        return np.Inf

    def adjust_weights(self, scale):
        if self.vertical:
            self.w_jk[1][0] = scale
        if self.horizontal:
            self.w_jk[0][1] = scale
        if self.diagonal:
            self.w_jk[1][1] = scale
        if self.color:
            self.w_c[1] = scale
