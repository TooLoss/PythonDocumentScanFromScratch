import numpy as np
from numpy import ndarray

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
    return D[n]/(x*y)

def FADIT(I, p: (int, int) = 0):
    """
    Donne un seuil pour l'image I noir et blanc
    @param I: Image
    @param p: Padding de l'image pour l'analyse du seuil
    @return: L'image seuillé
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

def NormeLocale(M, C:(int, int), n=1):
    '''
    M : Image
    C : Taille de la plage
    n : norme 1, 2 ...
    return l'intégrale dans la plage [0,x], [0,y]
    '''
    x, y = C
    mx, my = M.shape
    I = 0
    for i in range(x):
        for j in range(y):
            if x<mx and y<my:
                I += M[i,j]**n
    return I

def MoyenneRectange(M, C, T):
    '''
    M : Image
    n : Moyenne 1, 2, ...
    C : Position du pixel
    T : Taille du rectangle 
    '''
    x,y = C
    dx,dy = T
    maxx, maxy = M.shape
    if x-dx<0 or y-dy<0 or x+dx>=maxx-1 or y+dy>=maxy-1:
        dx = min(maxx-x-1, x)
        dy = min(maxy-y-1, y)
    N = (2*dx+1)*(2*dy+1)
    m = (NormeLocale(M,(x+dx,y+dy),1) + NormeLocale(M,(x-dx,y-dy),1)
         - NormeLocale(M,(x-dx,y-dy),1) - NormeLocale(M,(x+dx,y-dy),1))*(1/N)
    return m

def VarianceRectangle(M, C:(int,int), T:(int,int)):
    '''
    M : Image
    (x,y) : Position du pixel
    (dx,dy) : Taille du rectangle 
    '''
    x,y = C
    dx,dy = T
    N = (2*dx+1)*(2*dy+1)
    S = (NormeLocale(M,(x+dx,y+dy),2) + NormeLocale(M,(x-dx,y-dy),2)
         - NormeLocale(M,(x+dx,y-dy),2) - NormeLocale(M,(x+dx,y-dy),2))*1/(N-1) - (1/N)*(MoyenneRectange(M,(x,y),(dx,dy))*N)**2
    return S

def NiblackParam(I, k, S):
    x,y = I.shape
    P = []
    for i in range(x):
        for j in range(y):
            m = MoyenneRectange(I, (i,j), (S,S))
            v = VarianceRectangle(I, (i,j), (S,S))
            P.append((m, v))
    return P

def Erosion(I, P:(int,int), T:(int,int)):
    pass

def ErosionElementaire(I, P:(int,int)):
    mx, my = I.shape
    x, y = T
    a, b = P
    cx = (x-1)//2
    cy = (y-1)//2
    S = 0
    for i in range(x):
        for j in range(y):
            if (0 <= (a - cx) + i < mx) and (0 <= (b - cy) + j < my):
                if I[a-cx+i, b-cy+j] > 10:
                    return False
