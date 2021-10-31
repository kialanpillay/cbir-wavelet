import numpy as np
import pywt
from sklearn.decomposition import PCA

from utils import resize


def preprocess(img):
    """
    Preprocesses an image through rescaling and color-space remapping

    Parameters
    ----------
    img : numpy.ndarray
       Image data

    Returns
    -------
    numpy.ndarray
    """
    return remap_color(rescale(img))


def rescale(img):
    """
    Wrapper function to resize an image to 128 x 128 dimensions

    Parameters
    ----------
    img : numpy.ndarray
       Image data

    Returns
    -------
    numpy.ndarray
    """
    return resize(img)


def remap_color(img):
    """
    Remaps an RGB image to the C1, C2, C3 opponent colour-space.
    Returns a stacked image with three channels.

    Parameters
    ----------
    img : numpy.ndarray
       Image data

    Returns
    -------
    numpy.ndarray
    """
    MAX = np.amax(img)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    C1 = (R + G + B) / 3
    C2 = (R + (MAX - B)) / 2
    C3 = (R + 2 * (MAX - G) + B) / 4
    return np.dstack((C1, C2, C3))


def wavelet_transform(c, level):
    """
    Wrapper function to compute the DWT

    Parameters
    ----------
    c : numpy.ndarray
       C1, C2, C3 image data
    level : int
       Wavelet transform decomposition level

    Returns
    -------
    numpy.ndarray
    """
    return fwt(c, level)


def fwt(c, level=3):
    """
    Computes a wavelet descriptor for an image using 2D DWT.

    Parameters
    ----------
    c : numpy.ndarray
       C1, C2, C3 image data
    level : int, optional
       Wavelet transform decomposition level

    Returns
    -------
    numpy.ndarray
    """
    WC1, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 0], wavelet='db4', level=level, mode='periodization'))
    WC2, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 1], wavelet='db4', level=level, mode='periodization'))
    WC3, _ = pywt.coeffs_to_array(pywt.wavedec2(c[:, :, 2], wavelet='db4', level=level, mode='periodization'))
    return np.dstack((WC1, WC2, WC3))


def generate_feature(w, w_, pca=False):
    if pca:
        return dimensionality_reduction(
            np.array([compute_standard_deviation(w), extract_submatrix(w_, dim=(8, 8)), extract_submatrix(w)],
                     dtype=object))
    else:
        return np.array([compute_standard_deviation(w), extract_submatrix(w_, dim=(8, 8)), extract_submatrix(w)],
                        dtype=object)


def extract_submatrix(w, dim=(16, 16)):
    """
    Extracts a wavelet transform coefficient matrix of the specified dimensions.

    Parameters
    ----------
    w : numpy.ndarray
       WC1, WC2, WC3 image data
    dim : (int, int), optional
       Submatrix dimensions

    Returns
    -------
    numpy.ndarray
    """
    S1 = w[0:dim[0], 0:dim[1], 0]
    S2 = w[0:dim[0], 0:dim[1], 1]
    S3 = w[0:dim[0], 0:dim[1], 2]
    return np.dstack((S1, S2, S3))


def compute_standard_deviation(w):
    """
    Computes the standard deviation of a coefficient submatrix

    Parameters
    ----------
    w : numpy.ndarray
       WC1, WC2, WC3 image data

    Returns
    -------
    numpy.ndarray
    """
    return np.array(
        [np.std(w[0:8, 0:8, 0], dtype=np.float64), np.std(w[0:8, 0:8, 1], dtype=np.float64), np.std(w[0:8, 0:8, 2],
                                                                                                    dtype=np.float64)])


def match(s, s_, beta=0.5):
    """
    Query and candidate image feature standard deviation similarity match

    Parameters
    ----------
    s : numpy.ndarray
       Query image standard deviations
    s_ : numpy.ndarray
       Candidate image standard deviations
    beta: float, optional
        Threshold

    Returns
    -------
    bool
    """
    if (s[0] * beta < s_[0] < s[0] / beta) or (
            (s[1] * beta < s_[1] < s[1] / beta) and (s[2] * beta < s_[2] < s[2] / beta)):
        return True
    else:
        return False


def dist(w, w_, weights_jk=np.array([[1, 1], [1, 1]]), weights_c=np.array([1, 1, 1])):
    """
    Computes the standard deviation of a coefficient submatrix

    Parameters
    ----------
    w : numpy.ndarray
       Query image feature vector
    w_ : numpy.ndarray
       Candidate image feature vector
    weights_jk: numpy.ndarray
        Distance measure detail weights
    weights_c: numpy.ndarray
        Distance measure channel weights

    Returns
    -------
    float
    """
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


def dimensionality_reduction(f, n_components=2):
    """
    Reduces the dimensionality of an image feature vector

    Parameters
    ----------
    f : numpy.ndarray
       Image feature vector
    n_components : int, optional
       Principal components

    Returns
    -------
    numpy.ndarray
    """
    pca = PCA(n_components=n_components)
    for i in [1, 2]:
        f_ = np.zeros(shape=(n_components, f[i].shape[1], f[i].shape[2]))
        for d in range(f[i].shape[2]):
            pca.fit(f[i][:, :, d])
            f_[:, :, d] = pca.components_
        f[i] = f_
    return f
