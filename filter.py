import numpy as np
from numba import jit, njit
import main as m


def WhiteScale(I, p):
    x, y = I.shape
    low, high = I.min()/p, I.max()/p
    for i in range(x):
        for j in range(y):
            I[i,j] = 255*(I[i,j]-low)/(high-low)
    return I


def Unsharp(I, f=1):
    """
    Fonction netteté
    :param I: matrice image
    :param f: facteur de netteté, 0 pour rien
    """
    Filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * f
    return m.Convolution(I, Filter)


def Moyenne(L, P):
    moy = 0
    for n in range(L):
        moy += n*P(n)
    return moy


def DicoPixelCount(I):
    PixelCount = {x:0 for x in range(int(np.max(I) + 1))}
    x, y = I.shape
    for i in range(x):
        for j in range(y):
            PixelCount[I[i,j]] += 1
    return PixelCount


GlobalDico = {}
CacheImage = np.zeros((1,1))


def DicoProba(n, D=None):
    if D is None:
        D = GlobalDico
    x, y = np.shape(CacheImage)
    if n in D:
        return D[n]/(x*y)
    else:
        return 0


def FADIT(I, p: (int, int) = (0,0)):
    """
    Donne un seuil pour l'image I noir et blanc
    :param I: Image
    :param p: Padding de l'image pour l'analyse du seuil
    :return: L'image seuillé
    """
    Img = np.abs(np.floor(I).astype(int))
    global CacheImage
    global GlobalDico
    CacheImage = Img
    GlobalDico = DicoPixelCount(m.Padding(Img, p))
    C = Criterion(DicoProba, np.max(Img))
    print("Seuil : " + str(C))
    return m.Seuil(Img, C)


def Criterion(P, L=255):
    """
    Fonction criterion pour une probabilité donnée
    """
    t_max = 0
    C_max = 0
    for t in range(L):
        S_t = Pi(P, t)
        moy = Moyenne(L, P)
        f_t = moy / (moy + (t * (t + 1) / 2) * (1 - (moy / (L - 1))))
        C_t = 2*S_t*f_t - S_t - f_t + 1
        if C_t > C_max:
            C_max = C_t
            t_max = t
    return t_max


def Pi(P, t):
    S = 0
    for n in range(t):
        S += P(n)
    return S


def Dilatation(I, r = 1):
    """
    Dilate l'image
    :param I: Image
    :param r: Rayon dilatation.
    :return: Image dilaté
    """
    x, y = I.shape
    R = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            L = m.Surrouding(I, (i,j), r)
            if 0 not in L:
                R[i,j] = 255
    return R


def MedianBlur(I, r):
    """
    Prend les pixels voisins pour chaque pixel le remplace par la medianne.
    :param I: Image
    :param r: Rayon
    :return: Image mediane
    """
    x, y = I.shape
    R = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            L = m.Surrouding(I, (i,j), r)
            median = np.mean(L)
            if I[i,j] > 0:
                R[i,j] = median*.5
    return R


@njit
def Sauvola(I, r, k=0.2, p=128):
    """
    Renvoie la matrice avec le seuil appliqué avec la méthode Sauvola.
    :param I: Image
    :param r: Taille du voisinage pris en compte
    :param k: Valeur empirique, importance de l'écart type. Généralement 0.2 donne un bon résultat.
    :param p: Normalise l'écart type. Si les pixels sont entre [0,255], R=128
    :return: matrice image seuillé.
    """
    x, y = I.shape
    P = np.zeros((x+2*r, y+2*r), dtype=I.dtype)
    P[r:-r, r:-r] = I # Initialise P qui contient I avec les marges étendues de r pour ne pas avoir index error
    R = np.zeros_like(I) # zeros_like permet d'être compatible avec Numba
    for i in range(x):
        for j in range(y):
            N = P[i:i+r, j:j+r]
            mean = np.mean(N) # Moyenne
            std = np.std(N) # Ecart-type
            treshold = mean * (1 + k*((std/p) - 1))
            if I[i,j] < treshold:
                R[i,j] = 0
            else:
                R[i,j] = 255
    return R