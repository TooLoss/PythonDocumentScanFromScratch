import numpy as np

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as math


def ProduitMatricielElementaire(A, B, i, j):
    '''
    Produit Matriciel Elementaire
    Opération (A*B) aux coordonnées i, j
    '''
    p = A.shape[0]
    S = 0
    for k in range(p):
        S += A[i, k] * B[k, j]
    return S


def ProduitMatriciel(A, B):
    '''
    Fait le produit matriciel sur toute la matrice
    '''
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    j = B.shape[1]
    R = np.zeros((n, j), float)
    for i in range(n):
        for j in range(j):
            R[i, j] = ProduitMatricielElementaire(A, B, i, j)
    return R

def Produit(A, B):
    '''
    Produit coordonnée à coordonnée de la matrice
    '''
    n = A.shape[0]
    p = B.shape[1]
    R = np.zeros((n, p), float)
    for i in range(n):
        for j in range(p):
            R[i,j] = A[i,j] * B[i,j]
    return R

def ConvolutionElementaire(A, a, b, C):
    '''
    Fait le produit de la matrice. Version centré
    '''
    mx, my = A.shape
    x, y = C.shape
    cx = (x-1)//2
    cy = (y-1)//2
    S = 0
    for i in range(x):
        for j in range(y):
            if ((a-cx)+i >= 0 and (a-cx)+i < mx) and ((b-cy)+j >= 0 and (b-cy)+j < my):
                S += A[a-cx+i, b-cy+j] * C[i, j]
    return S


def Convolution(A, C):
    '''
    Convolution A*C
    '''
    x, y = A.shape
    R = np.zeros((x, y), A.dtype)
    for i in range(x):
        for j in range(y):
            R[i, j] = ConvolutionElementaire(A, i, j, C)
    return R


def GaussianFilter(n, s):
    '''
    n : Taille du filtre gaussien
    s : Coefficient
    '''
    G = np.zeros((n, n))
    M = (n+1)//2
    for i in range(n):
        for j in range(n):
            G[i, j] = (1/2*np.pi*(s**2)) * \
                np.exp(-((i-(M-1))**2 + (j-(M-1))**2)/(2 * s**2))
    return G/np.sum(G) # Normalise pour conserver la même luminosité 


def GaussianBlur(A, n, s):
    '''
    Applique un filtre Gaussian sur l'image
    n : Taille du filtre
    s : Paramètre
    '''
    return Convolution(A, GaussianFilter(n, s))


def Canny(M, CX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), CY=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).transpose()):
    MX = Convolution(M, CX)
    MY = Convolution(M, CY)
    return MX, MY


def SquareMatrix(n, r, v):
    M = np.zeros((n, n), int)
    m = n//2
    k = r//2
    for i in range(r):
        for j in range(r):
            M[m-k+i, m-k+j] = v
    return M


def Compresser(M, f):
    '''
    Compresse l'image d'un facteur f
    '''
    x, y = M.shape
    I = np.zeros((x//f, y//f))
    for i in range(x//f):
        for j in range(y//f):
             I[i, j] = M[i*f, j*f]
    return I


def Afficher(M, min=0, max=255):
    plt.imshow(M, cmap='gray', vmin=min, vmax=max)
    plt.show()


def NormeAngle(MX, MY):
    x, y = MX.shape
    Theta = np.arctan2(MY, MX) * 180/np.pi
    Norme = np.sqrt(MX**2 + MY**2)
    return Theta, Norme

def NonMaximum(A, N):
    '''
    A matrice d'angle
    N norme
    '''
    x, y = A.shape
    R = N.copy()
    for i in range(1, x-1):
        for j in range(1, y-1):
            v = A[i,j]
            v = v if v >= 0 else v + 180
            # symétrie de 0 a 180 et 180 à 3600
            v = v if v <= 180 else 360-v
            if v < 22.5:
                # Pixels horizontaux 
                if N[i,j+1] >= N[i,j] or N[i,j-1] >= N[i,j]:
                    R[i,j] = 0
            elif v < 67.5:
                # Pixels diaagonaux 
                if N[i-1,j+1] >= N[i,j] or N[i+1,j-1] >= N[i,j]:
                    R[i,j] = 0
            elif v < 112.5:
                # Pixel verticaux 
                if N[i+1,j] >= N[i,j] or N[i-1,j] >= N[i,j]:
                    R[i,j] = 0
            elif v < 157.5:
                # Pixels diagonaux 
                if N[i+1,j+1] >= N[i,j] or N[i-1,j-1] >= N[i,j]:
                    R[i,j] = 0
            else:
                # Pixel horizontaux autre côté 
                if N[i,j+1] >= N[i,j] or N[i,j-1] >= N[i,j]:
                    R[i,j] = 0
    return R

def Bornes(M):
    '''
    Renvoie la valeur minimale et maximale de la matrice
    '''
    x, y = M.shape
    min = M[0,0]
    max = M[0,0]
    for i in range(x):
        for j in range(y):
            if M[i,j] > max:
                max = M[i,j]
            if M[i,j] < min:
                min = M[i,j]
    return min, max

def Histogramme(M, p):
    '''
    Crée un histogramme des valeurs
    p : nombre après la virgule, 0 par défaut
    '''
    H = {p*i:0 for i in range(-255, 255*255//p)}
    x, y = M.shape
    for i in range(x):
        for j in range(y):
            H[int(M[i,j]//p * p)] += 1
    return H

def Hysteresis(M, bas = 0.05, haut = 0.09):
    min, max = Bornes(M)
    hist = Histogramme(M, 1)
    pas = (max - min)/len(hist)
    valeurs = np.linspace(start, stop)

def Seuil(M, s):
    x, y = M.shape
    for i in range(x):
        for j in range(y):
            if M[i,j] < s:
                M[i,j] = 0
    return M

def MinDistanceList(c, L):
    if len(L) == 0:
        return 0
    min = x**2 + y**2
    for e in L:
        dist = np.sqrt((c[0]-e[0])**2 + (c[1] - e[1])**2)
        if dist < min:
            min = dist
    return min

def PointsRectangle(P, nbMax):
    x, y = P.shape
    coord = []
    for i in range(x):
        for j in range(y):
            if P[i,j] > 0 and MinDistList([x,y], coord) > (x+y)/4:
                coord.append([i,j])


