import os

import cv2


def read(dirname, filename):
    return cv2.imread(filename=str(os.path.join(dirname, filename)))


def resize(img, dim=(128, 128)):
    return cv2.resize(img, dim, cv2.INTER_LINEAR)


def write(dirname, filename, img):
    cv2.imwrite(filename=str(os.path.join(dirname, filename)), img=img)
