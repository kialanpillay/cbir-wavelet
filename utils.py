import os

import cv2

WINDOW_NAME = "CBIR"


def file_extension(filename, ext):
    """
    Concatenates a filename and extension

    Parameters
    ----------
    filename : str
       Filename
    ext : str
       File extension

    Returns
    -------
    str
    """
    return filename + "." + ext


def read(dirname, filename):
    """
    Wrapper function for cv2.imread

    Parameters
    ----------
    dirname : str
       Directory name
    filename : str
       Filename

    Returns
    -------
    numpy.array
    """
    return cv2.imread(filename=str(os.path.join(dirname, filename)))


def resize(img, dim=(128, 128)):
    """
    Resizes an image using the bilinear interpolation method

    Parameters
    ----------
    img : numpy.array, optional
        Image data
    dim : (int, int)
        Rescale dimensions
    Returns
    -------
    numpy.array
    """
    return cv2.resize(img, dim, cv2.INTER_LINEAR)


def write(dirname, filename, img=None):
    """
    Wrapper function for cv2.imwrite

    Parameters
    ----------
    dirname : str
       Directory name
    filename : str
       Filename
    img : numpy.ndarray, optional
        Image data
    """
    if img is not None:
        cv2.imwrite(filename=str(os.path.join(dirname, file_extension(filename, "jpg"))), img=img)
    else:
        img = read(dirname[0:len(dirname) - 4], file_extension(filename, "jpg"))
        cv2.imwrite(filename=str(os.path.join(dirname, file_extension(filename, "jpg"))), img=img)


def display(dirname, filename, delay=1500, img=None):
    """
    Wrapper function for cv2.imshow

    Parameters
    ----------
    dirname : str
       Directory name
    filename : str
       Filename
    delay : int, optional
         Viewer window millisecond delay
    img : numpy.ndarray, optional
        Image data
    """
    if img is not None:
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
    else:
        img = read(dirname, file_extension(filename, "jpg"))
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()
