from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as math
from numba import jit, njit, prange # Permet de faire du multithreading et d'accélerer le temps d'execution
from numba.typed import Dict


@njit
def ProduitMatricielElementaire(A, B, i, j):
    """
    Produit Matriciel Elementaire
    Opération (A*B) aux coordonnées i, j
    """
    p = A.shape[0]
    S = 0
    for k in range(p):
        S += A[i, k] * B[k, j]
    return S


@njit
def ProduitMatriciel(A, B):
    """
    Fait le produit matriciel sur toute la matrice
    """
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    j = B.shape[1]
    R = np.zeros((n, j), float)
    for i in range(n):
        for j in range(j):
            R[i, j] = ProduitMatricielElementaire(A, B, i, j)
    return R


@njit
def Produit(A, B):
    """
    Produit coordonnée à coordonnée de la matrice
    """
    n = A.shape[0]
    p = B.shape[1]
    R = np.zeros((n, p), float)
    for i in range(n):
        for j in range(p):
            R[i,j] = A[i,j] * B[i,j]
    return R


@njit
def ConvolutionElementaire(A, a, b, C):
    """
    Fait le produit de la matrice. Version centré
    """
    mx, my = A.shape
    x, y = C.shape
    cx = (x-1)//2
    cy = (y-1)//2
    S = 0
    for i in range(x):
        for j in range(y):
            if (0 <= (a - cx) + i < mx) and (0 <= (b - cy) + j < my):
                S += A[a-cx+i, b-cy+j] * C[i, j]
    return S


@njit
def Convolution(A, C):
    """
    Convolution A*C
    """
    x, y = A.shape
    R = np.zeros((x, y), A.dtype)
    for i in range(x):
        for j in range(y):
            R[i, j] = ConvolutionElementaire(A, i, j, C)
    return R


def GaussianFilter(n, s):
    """
    n : Taille du filtre gaussien
    s : Coefficient
    """
    G = np.zeros((n, n))
    M = (n+1)//2
    for i in range(n):
        for j in range(n):
            G[i, j] = (1/2*np.pi*(s**2)) * \
                np.exp(-((i-(M-1))**2 + (j-(M-1))**2)/(2 * s**2))
    return G/np.sum(G) # Normalise pour conserver la même luminosité 


def GaussianBlur(A, n, s):
    """
    Applique un filtre Gaussian sur l'image
    @param A: Image entrée
    @param n: Taille du filtre
    @param s: Puissance du flou [0, 1]
    @return: Image flouté
    """
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
    """
    Compresse l'image d'un facteur f
    """
    if f == 1:
        return M
    x, y = M.shape
    I = np.zeros((x//f, y//f))
    for i in range(x//f):
        for j in range(y//f):
             I[i, j] = M[i*f, j*f]
    return I


def AfficherAxe(M, min=0, max=255):
    plt.imshow(M, cmap='gray', vmin=min, vmax=max)
    plt.show()

def Afficher(M):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8.3, 11.7, 150) # A4
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(M, cmap='gray', interpolation='none')

def SaveAsPng(M, filename : str):
    """
    Sauvegarde l'image au format PNG
    :param M: Image
    :param filename: chemin absolue du fichier. Pour avoir le relatif, commencer par ./
    """
    if not filename.endswith('.png'):
        filename += '.png'
    Base = M.copy()
    rgba_image = np.zeros((*M.shape, 4), dtype=np.uint8)
    rgba_image[..., 0] = M  # Red channel
    rgba_image[..., 1] = M  # Green channel
    rgba_image[..., 2] = M  # Blue channel
    rgba_image[..., 3] = np.where(Base == -1, 0, 255)
    plt.imsave(filename, rgba_image)

def ImportAsPng(filename):
    I = Image.open(filename, mode="r").convert('RGBA')
    image_array = np.array(I)

    alpha = image_array[..., 3]
    Img = (0.299 * image_array[..., 0] +
           0.587 * image_array[..., 1] +
           0.114 * image_array[..., 2]).astype(np.float32)
    Img[alpha == 0] = -1
    return Img

def NormeAngle(MX, MY):
    x, y = MX.shape
    Theta = np.arctan2(MY, MX) * 180/np.pi
    Norme = np.sqrt(MX**2 + MY**2)
    return Theta, Norme

def NonMaximum(A, N):
    """
    Recherche maximums le long d'une ligne.
    :param A: Matrice qui contient l'angle (retourné de la fonction Canny).
    :param N: Matruce qui contient la norme du contour (retourné de la fonction Canny).
    :return: N mais débruité.
    """
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

def Histogramme(M, p):
    """
    Crée un histogramme des valeurs
    p : nombre après la virgule, 0 par défaut
    """
    H = {p*i:0 for i in range(-255, 255*255//p)}
    x, y = M.shape
    for i in range(x):
        for j in range(y):
            H[int(M[i,j]//p * p)] += 1
    return H

def Seuil(M, s):
    x, y = M.shape
    for i in range(x):
        for j in range(y):
            if M[i,j] < s:
                M[i,j] = 0
            else:
                M[i,j] = 255
    return M

def Padding(I, p : (int, int)):
    x, y = I.shape
    R = np.zeros((x-2*p[0], y-2*p[1]), dtype=int)
    for i in range(p[0], x-p[0]):
        for j in range(p[1], y-p[1]):
            R[i-p[0],j-p[1]] = I[i,j]
    return R

@njit
def Surrouding(I, pos : (int,int), r):
    L = []
    x, y = pos
    maxx, maxy = I.shape
    for i in range(x-r, x+r+1):
        for j in range(y-r, y+r+1):
            if 0 <= i < maxx and 0 <= j < maxy:
                L.append(I[i,j])
    return L

def Cut(M, A : (int, int), B : (int, int)):
    """
    Coupe l'image dans le rectangle A B
    :return: Image coupé
    """
    x = B[0] - A[0]
    y = B[1] - A[1]
    magin_x = A[0]
    magin_y = A[1]
    C = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            C[i,j] = M[i+magin_x,j+magin_y]
    return C


def ConnectedComponentLabeling(I, b = 0):
    """
    Algorithme de détection de blob.
    Retourn la positions des blobs de l'image.
    :param b: Background color
    :param I: Matrice image
    :return: La matrice avec les différentes classes d'équivalence
    """
    x, y = np.shape(I)
    Label = np.zeros((x, y), dtype=int)
    tag = 1
    Redef = {}
    for i in range(1, x-1):
        for j in range(1, y-1):

            LVoisin = [Label[i-1,j], Label[i,j-1]]
            LVoisin = [i for i in LVoisin if i != 0]

            if I[i,j] != b:
                if len(LVoisin) == 0: # Si on tombe sur une nouvelle classe
                    Label[i,j] = tag
                    tag += 1
                else: # Si il y a 1 ou plus de classes attachées.
                    min_tag = min(LVoisin)
                    Label[i, j] = min_tag
                    for n in LVoisin:
                        if n != min_tag:
                            Redef[n] = min_tag
                            # Si plus de 2 ou plus classes sont touchée, elles pointent toutes sur le min tag
    # Joindre les classes qui se touchent
    for i in range(x):
        for j in range(y):
            if Label[i, j] != 0:
                label = Label[i, j]
                while label in Redef: # On parcours le graphe pour arriver au minimum
                    label = Redef[label]
                Label[i, j] = label

    return Label


@njit
def AutoCrop(I, val=0):
    """
    Redimensionne l'image pour ne contenir que val
    :param I: Image
    :param val: valeur a isoler
    :return:
    """
    x, y = I.shape
    xmin, ymin, = x-1, y-1
    xmax, ymax = 0, 0
    for i in range(x):
        for j in range(y):
            if I[i, j] == val:
                if i < xmin:
                    xmin = i
                if j < ymin:
                    ymin = j
                if i > xmax:
                    xmax = i
                if j > ymax:
                    ymax = j
    C = I[xmin:xmax, ymin - 1:ymax + 1]
    return C, xmin, ymin, xmax, ymax