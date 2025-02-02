import numpy as np
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

def DicoPixelCount(I):
    PixelCount = {x:0 for x in range(int(np.max(I) + 1))}
    x, y = I.shape
    for i in range(x):
        for j in range(y):
            PixelCount[I[i,j]] += 1
    return PixelCount

GlobalDico = {}
CacheImage = np.zeros((1,1))

def DicoProba(n, D = GlobalDico):
    x, y = np.shape(CacheImage)
    return D[n]/(x*y)

def FADIT(I, p=0):
    """
    Donne un seuil pour l'image I noir et blanc
    I : Image
    p : Padding
    """
    Img = np.floor(I).astype(int)
    CacheImage = Img
    GlobalDico = DicoPixelCount(Img)
    C = Criterion(DicoProba, np.max(Img))
    return C


def Criterion(P, L=255):
    """
    Fonction criterion pour une probabilité donnée
    """
    t_max = 0
    C_max = 0
    for t in range(L):
        S_t = Pi(P, t)
        moy = Moyenne(L, P)
        f_t = moy / ((moy + t*(t+1)/2)*(1 - moy/L-1))
        C_t = 2*S_t*f_t - S_t - f_t + 1
        if C_t > C_max:
            C_max = C_t
            T_max = t
    return t_max

def Pi(P, t):
    S = 0
    for n in range(t):
        S += P(n)
    return S
