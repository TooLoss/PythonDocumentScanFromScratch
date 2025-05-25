import filter as f
import main as m
import numpy as np
from numba import jit, njit, prange


def CharacterList(I, mcount = 0):
    """
    Donne une liste de matrice représentant chaque caractères dans l'image.
    :param I: Matrice image
    :param mcount: Le nombre minumums de pixels pour considérer le caractère.
    :return:
    """
    x, y = I.shape
    L = m.ConnectedComponentLabeling(I, 255)
    l, count = np.unique(L, return_counts=True) # Donne la liste des occurences des classes et leurs nombres d'apparition
    D = dict(zip(l, count))
    m.Afficher(L)
    LCaracter = []

    for i in range(max(D.keys())):
        if (i in D) and (D[i] < mcount):
            D.pop(i)
    for classe in D.keys():
        C, xmin, ymin, xmax, ymax = m.AutoCrop(L, classe)
        LCaracter.append(((xmin, ymin), I[xmin:xmax, ymin+1:ymax+1]))
    return LCaracter


