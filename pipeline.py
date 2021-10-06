import numpy as np

from utils import resize


def preprocess(db):
    res = []
    for img in db:
        res.append(remap_color(rescale(img)))

    return res


def rescale(img):
    return resize(img)


def remap_color(img):
    MAX = 255
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    C1 = (R + G + B)/3
    C2 = (R + (MAX - B))/2
    C3 = (R + 2 * (MAX - G) + B)/4
    return np.dstack((C1, C2, C3))

