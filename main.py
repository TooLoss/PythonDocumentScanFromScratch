import numpy as np

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math as math


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
    @param M: Image
    @param filename: nome du fichier
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
    A matrice d'angle
    N norme
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
    R = np.zeros((x-2*p[0], y-2*p[1]))
    for i in range(p[0], x-p[0]):
        for j in range(p[1], y-p[1]):
            R[i-p[0],j-p[1]] = I[i,j]
    return R
