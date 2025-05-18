import numpy as np
import main as m
from numba import jit
import matplotlib.pyplot as plt

# TODO Faire une fonction pour normaliser les coordonnées

def Normalize(points):
    centroid = np.mean(points, axis=0)
    distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
    mean_distance = np.mean(distances)
    scale = np.sqrt(2) / mean_distance
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    normalized_points = np.dot(T, np.vstack((points.T, np.ones(points.shape[0]))))

    return normalized_points.T, T


@jit(nopython=True)
def ComputeHomography(PE, PS):
    """
    PE : Points d'entrée
    PS : Points de sortie
    """
    assert len(PE) == len(PS), "Le nombre de points d'entrée doit être égal au nombre de points de sorties"
    assert len(PE) >= 4, "Pour résoudre la projection, il faut un mimumum de 4 points"

    A = []
    B = []
    for i in range(len(PE)):
        xe, ye = PE[i][0], PE[i][1]
        xs, ys = PS[i][0], PS[i][1]
        A.append([xe, ye, 1, 0, 0, 0, -xe*xs, -ye*xs])
        B.append([xs])
        A.append([0, 0, 0, xe, ye, 1, -xe * ys, -ye * ys])
        B.append([ys])
    A = np.array(A)
    B = np.array(B)

    h = np.dot(np.linalg.inv(A), B) # On calcule la matrice inverse H
    H = np.array([[h[0][0], h[1][0], h[2][0]],
                  [h[3][0], h[4][0], h[5][0]],
                  [h[6][0], h[7][0], 1]
                 ])

    return H


def EstDansCube(x, C):
    """
    Revoie vrai si x est dans le cube dont les coordonnées sont dans C
    """
    return (C[0,0] >= x[0] >= C[0,1]) and (C[1,0] >= x[0] >= C[0,1])


def ErrorNonAffected(I, C):
    x = C[1][0] - C[0][0]
    y = C[3][0] - C[2][0]
    error = 0
    for i in range(x):
        for j in range(y):
            if I[i+C[0][0],j+C[2][0]] == 0:
                error += 1
    return error


@jit(nopython=True)
def InterpolationRound(Point, S):
    return round(Point[0]), round(Point[1])


@jit(nopython=True)
def InterpolationNeighbour(Point, S):
    # TODO Problème si la sortie est trop petite, l'erreur est en cascade
    x, y = InterpolationRound(Point, S)
    mx, my = S.shape
    if not(2 < x < mx-2 and 2 < y < my-2): # Néglige les bords
        return x, y
    if S[x][y] < 0:
        return x, y
    # Cas où il y a déjà un pixel
    for i in range(-1, 2):
        for j in range(-1, 2):
            if S[x+i][y+j] < 0:
                return x+i, y+j
    return x, y

@jit(nopython=True)
def TransformationProjective(I, PE, PS, Interpolation : callable = InterpolationRound):
    h, w = I.shape
    S = np.full((w, h), -1)
    matrix = ComputeHomography(PE, PS).astype(np.float64)
    error_count = 0

    for i in range(h):
        for j in range(w):
            vec =  np.array([j, i, 1], dtype=np.float64)
            tmp = matrix @ vec
            x, y = tmp[0]/tmp[2], tmp[1]/tmp[2]
            x, y = Interpolation([x, y], S) # Méthode d'interpolation
            if 0 < x < w and 0 < y < h:
                # Calcul de l'erreur à retirer
                if S[x][y] != -1:
                    error_count += 1
                S[x][y] = I[i][j]
    return S.transpose(), error_count


def FillBasedOnNeighbour(M):
    x, y = M.shape
    Img = M.copy()
    for i in range(x):
        for j in range(y):
            if M[i][j] == -1:
                Img[i, j] = CloneNeighbour(M, (i, j), 1)
    return Img


def CloneNeighbour(M, P, r=1):
    x, y = P
    Points = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if 0 <= x+i < M.shape[0] and 0 <= y+j < M.shape[1]:
                if  M[x+i][y+j] >= 0:
                    Points.append(M[x+i, y+j])
    return MajorityPoint(Points)


def MajorityPoint(Points):
    if len(Points) == 0:
        return 0
    m = np.mean(Points)
    d = {abs(Points[i]-m): i for i in range(len(Points))}
    vmin = np.min(np.abs(Points - m))
    return Points[d[vmin]]


def CropImage(H, PS = [[2408,0],[2408,3508],[0,0],[0,3508]]):
    """
    H : Image transformed
    PS : Image Coordinates
    """
    Doc = H[round(PS[2][1]):round(PS[1][1]), round(PS[2][0]):round(PS[1][0])]
    return Doc

def ConnectedComponentLabeling(I):
    """
    Algorithme de détection de blob.
    Retourn la positions des blobs de l'image.
    :param I: Matrice image
    :return: La matrice avec les différentes classes d'équivalence
    """
    x, y = np.shape(I)
    Label = np.zeros((x, y), dtype=int)
    tag = 1
    Redef = {}
    for i in range(1, x-1):
        for j in range(1, y-1):
            L = [Label[i+1,j], Label[i,j+1], Label[i-1,j], Label[i,j-1]] # Crée la list de type int
            L = [i for i in L if i != 0]
            mtag = 0
            if len(L) > 0:
                mtag = min(L)
            if I[i,j] > 0 and Label[i,j] == 0:
                if mtag == 0:
                    Label[i,j] = tag
                    tag += 1
                elif (mtag == Label[i - 1,j] or Label[i - 1,j] == 0) and (mtag == Label[i, j - 1] or Label[i, j - 1] == 0):
                    Label[i,j] = mtag
                else: # Conflits entre 2 classes ou plus, donc on les ajoute
                    for e in L:
                        if e != mtag:
                            Redef[e] = mtag
                    Label[i,j] = mtag
    for i in range(x):
        for j in range(y):
            if Label[i,j] in Redef.keys():
                Label[i,j] = Redef[Label[i,j]]
    return Label


def SortConvention(L):
    """
    Ordonne la liste selon la convention HG, HD, BD, BG
    """
    assert len(L)==4, "Error : SortConvention. Il faut 4 points. " + str(len(L)) + " sont données."
    R = []
    I = {}
    X, Y = zip(*L)
    x_avg = np.mean(X)
    y_avg = np.mean(Y)
    for i in range(len(L)):
        if L[i][0] < x_avg and L[i][1] < y_avg: # Haut Gauche
            I[2] = i
        elif L[i][0] > x_avg and L[i][1] < y_avg: # Haut Droite
            I[0] = i
        elif L[i][0] > x_avg and L[i][1] > y_avg: # Bas Droite
            I[1] = i
        elif L[i][0] < x_avg and L[i][1] > y_avg: # Bas Gauche
            I[3] = i
    for i in range(4):
        R.append(L[I[i]])
    return R


def Dist2D(A, B):
    return np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


def MergeByDistance(P, d):
    for i in range(len(P)):
        R = []
        for j in range(i+1, len(P)):
            if Dist2D(P[i], P[j]) < d:
                R.append(P[j])
        for r in R:
            P.remove(r)


def BarycentreClasses(Label):
    """
    Utilise la fonction ConnectedComponentLabeling
    Retourne les positions des baricentres des différentes classes
    :param Label: Matrice label de la fonction
    :return:
    """
    D = {}
    x, y = Label.shape
    for i in range(x):
        for j in range(y):
            if Label[i,j] != 0:
                if Label[i,j] not in D:
                    D[Label[i,j]] = (i,j)
                else:
                    bx, by = D[Label[i, j]]
                    D[Label[i,j]] = ((bx+i)/2, (by+j)/2)
    return list(D.values())


def Hetzienne(x, y, Dx, Dy):
    DDX = 0
    DDY = 0
    DDXY = 0
    for i in range(x-1,x+1):
        for j in range(y-1,y+1):
            DDX += Dx[i,j]**2
            DDY += Dy[i,j]**2
            DDXY += Dx[i,j]*Dy[i,j]
    H = np.array([[DDX, DDXY], [DDXY, DDY]])
    return H


def HarrisCorner(M, k = .04):
    Ix, Iy = m.Canny(M)
    x, y = M.shape
    P = []
    for i in range(1, x-1):
        for j in range(1, y-1):
            H = Hetzienne(i, j, Ix, Iy)
            C = np.linalg.det(H) - k * np.trace(H)**2
            if C > 0:
                P.append((i,j))
    return P


def FindCornersPosition(I):
    """
    Cherche les coins de l'image
    :param I: Matrice image
    :return: Liste de tuple des coordonnées des coins
    """
    VX, VY = m.Canny(I)
    VX = m.GaussianBlur(VX, 10, 50)
    VY = m.GaussianBlur(VY, 10, 50)
    C = m.Produit(abs(VX), abs(VY))
    P = m.Seuil(C, np.max(C)*.5) # Matrice qui contient les 4 points
    m.Afficher(P)
    return BarycentreClasses(ConnectedComponentLabeling(P))


@jit(nopython=True)
def ContrastMax(I):
    x, y = I.shape
    for i in range(x):
        for j in range(y):
            if I[i, j] > 100:
                I[i, j] = 255
            else:
                I[i, j] = 0
    return I


def ToMatrix(P):
    R = np.zeros((len(P), 2))
    for i in range(len(P)):
        R[i,1] = P[i][0]
        R[i,0] = P[i][1] # Inversion due a la fonction TransformationProjective
    return R


def MakeHomographie(M):
    """
    Fait tout le procédé homographie automatiquement.
    :param M: Matrice image noir et blanc (Brut non compressé)
    :return: Image avec la transformation terminé
    """
    R = np.copy(M)
    M = m.GaussianBlur(M,10, 1)
    ContrastMax(M)
    P = FindCornersPosition(m.Compresser(M, 5))

    if len(P) > 4:
        MergeByDistance(P, M.shape[0]/10)

    P = [(x*5, y*5) for x, y in P] # Renormalise les points
    assert len(P) == 4, "Error : MakeHomographie. Les 4 côtées de l'image ne sont pas trouvé."

    P = SortConvention(P)
    PS = np.array([[0,3508],[2408,3508],[0,0],[2408,0]]) # Format A4
    AirePS = 2408*3508
    X, Y = zip(*P)
    AirePE = (max(X)-min(X))*(max(Y)-min(Y))
    PS = np.sqrt(AirePE/AirePS) * PS * .65 # Normalise PS, le facteur 0.8 permet de s'assurer qu'on préfère la surafectation à la non affectation
    P = ToMatrix(P)

    XS, YS = zip(*PS)
    XE, YE = zip(*P)

    print(P)
    print(PS)

    plt.scatter(XE, YE, color = 'r')
    plt.scatter(XS, YS, color = 'b')
    plt.imshow(R)
    plt.show()

    HImage, e_suraffectations = TransformationProjective(R, P, PS)
    H = m.Cut(HImage, (0,0), (int(PS[1,1]),int(PS[1,0])))
    return H.transpose()