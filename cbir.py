import numpy as np
import pywt

from utils import resize


def preprocess(img):
    return remap_color(rescale(img))


def rescale(img):
    return resize(img)


def remap_color(img):
    MAX = np.amax(img)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    C1 = (R + G + B) / 3
    C2 = (R + (MAX - B)) / 2
    C3 = (R + 2 * (MAX - G) + B) / 4
    return np.dstack((C1, C2, C3))


def wavelet_transform(c, level):
    return fwt(c, level)


def fwt(c, level=3):
    WC1, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 0], wavelet='db8', level=level, mode='periodization'))
    WC2, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 1], wavelet='db8', level=level, mode='periodization'))
    WC3, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 2], wavelet='db8', level=level, mode='periodization'))
    return np.dstack((WC1, WC2, WC3))


def generate_feature(w, w_):
    return np.array([extract_submatrix(w_, dim=(8, 8)), extract_submatrix(w), compute_standard_deviation(w)],
                    dtype=object)


def extract_submatrix(w, dim=(16, 16)):
    if dim == (8, 8):
        S1 = w[0:8, 0:8, 0]
        S2 = w[0:8, 0:8, 1]
        S3 = w[0:8, 0:8, 2]
    else:
        S1 = np.vstack([np.hstack((w[0:8, 0:8, 0], w[0:8, 8:16, 0])), np.hstack((w[8:16, 0:8, 0], w[8:16, 8:16, 0]))])
        S2 = np.vstack([np.hstack((w[0:8, 0:8, 1], w[0:8, 8:16, 1])), np.hstack((w[8:16, 0:8, 1], w[8:16, 8:16, 1]))])
        S3 = np.vstack([np.hstack((w[0:8, 0:8, 2], w[0:8, 8:16, 2])), np.hstack((w[8:16, 0:8, 2], w[8:16, 8:16, 2]))])
    return np.dstack((S1, S2, S3))


def compute_standard_deviation(w):
    return np.array(
        [np.std(w[0:8, 0:8, 0], dtype=np.float64), np.std(w[0:8, 0:8, 1], dtype=np.float64), np.std(w[0:8, 0:8, 2],
                                                                                                    dtype=np.float64)])


def match(s, s_, beta=0.5):
    if (s[0] * beta < s_[0] < s[0] / beta) or (
            (s[1] * beta < s_[1] < s[1] / beta) and (s[2] * beta < s_[2] < s[2] / beta)):
        return True
    else:
        return False


def dist(w, w_, weights_jk=np.array([[1, 1], [1, 1]]), weights_c=(1, 1, 1)):
    D1 = 0
    for i in range(3):
        D1 += weights_c[i] * np.linalg.norm(w[0:8, 0:8, i] - w_[0:8, 0:8, i])
    D2 = 0
    for i in range(3):
        D2 += weights_c[i] * np.linalg.norm(w[0:8, 8:16, i] - w_[0:8, 8:16, i])
    D3 = 0
    for i in range(3):
        D3 += weights_c[i] * np.linalg.norm(w[8:16, 0:8, i] - w_[8:16, 0:8, i])
    D4 = 0
    for i in range(3):
        D4 += weights_c[i] * np.linalg.norm(w[8:16, 8:16, i] - w_[8:16, 8:16, i])
    return weights_jk[0, 0] * D1 + weights_jk[0, 1] * D2 + weights_jk[1, 0] * D3 + weights_jk[1, 1] * D4
