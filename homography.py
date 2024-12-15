import numpy as np

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


def InterpolationRound(Point, S):
    return round(Point[0]), round(Point[1])

def InterpolationNeighbour(Point, S):
    # TODO Problème si la sortie est trop petite, l'erreur est en cascade
    x, y = InterpolationRound(Point, S)
    mx, my = S.shape
    if not(2 < x < mx-2 and 2 < y < my-2): # Néglige les bords
        return x, y
    if S[x][y] == -1:
        return x, y
    # Cas où il y a déjà un pixel
    for i in range(-1, 2):
        for j in range(-1, 2):
            if S[x+i][y+j] == -1:
                return x+i, y+j
    return x, y

def TransformationProjective(I, PE, PS, Interpolation : callable = InterpolationRound):
    h, w = I.shape
    S = np.full((w, h), -1)
    matrix = ComputeHomography(PE, PS)
    error_count = 0

    for i in range(h):
        for j in range(w):
            tmp = np.dot(matrix, np.array([[j, i, 1]]).T)
            x, y = tmp[0][0]/tmp[2][0], tmp[1][0]/tmp[2][0]
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
                if  M[x+i][y+j] != -1:
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