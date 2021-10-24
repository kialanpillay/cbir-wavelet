import os

import cv2


def file_extension(filename, ext):
    return filename + "." + ext


def read(dirname, filename):
    return cv2.imread(filename=str(os.path.join(dirname, filename)))


def resize(img, dim=(128, 128)):
    return cv2.resize(img, dim, cv2.INTER_LINEAR)


def write(dirname, filename, img=None):
    if img is not None:
        cv2.imwrite(filename=str(os.path.join(dirname, file_extension(filename, "jpg"))), img=img)
    else:
        img = read(dirname[0:len(dirname)-4], file_extension(filename, "jpg"))
        cv2.imwrite(filename=str(os.path.join(dirname, file_extension(filename, "jpg"))), img=img)
