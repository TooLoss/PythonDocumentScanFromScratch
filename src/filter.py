import numpy as np
from numba import njit
from src import main as m


@njit
def dilatation(I, radius=1, searchColor=0, filledColor=255):
    """
    Dilates an image by expanding where the specified color searchedColor is present within a neighborhood of radius.
    The dilated area is then filled with filledColor.
    :param I: Image
    :param searchColor: Target color value to check for in the neighborhood.
    :param radius: Radius of the neighborhood.
    :param filledColor: Color of the dilated area.
    :return: Dilated output image
    """
    x, y = I.shape
    R = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            L = m.get_surrounding(I, (i, j), radius)
            if searchColor in L:
                R[i,j] = filledColor
    return R

# tresholding function

def treshold_with_formula(I, radius, k, Formula):
    x, y = I.shape
    P = np.zeros((x + 2 * radius, y + 2 * radius), dtype=I.dtype)
    P[radius:-radius, radius:-radius] = I  # Initialise P qui contient I avec les marges étendues de r pour ne pas avoir index error
    R = np.zeros_like(I)  # zeros_like permet d'être compatible avec Numba
    for i in range(x):
        for j in range(y):
            N = P[i:i + radius, j:j + radius]
            mean = np.mean(N)  # Moyenne
            std = np.std(N)  # Ecart-type
            treshold = Formula(mean, std, k)
            if I[i, j] < treshold:
                R[i, j] = 0
            else:
                R[i, j] = 255
    return R


def Sauvola(I, r, k=0.2, p=128):
    def sauvola_fomula(mean, std, k):
        return mean * (1 + k * ((std / p) - 1))

    return treshold_with_formula(I, r, k, sauvola_fomula)


def Niblack(I, r, k):
    """
    Niblack is a local tresholding function.
    Niblack is more prone to noise.
    If the region is only white, then std is low and the treshold = average.
    :param I: Image
    :param r: Neighboor Radius
    :param k: Standart deviation parameter (empirical)
    :return: Tresholded output image
    """
    def niblack_fomula(mean, std, k):
        return mean + k * std

    return treshold_with_formula(I, r, k, niblack_fomula)