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


# TODO Revoir la position des pixels pour éviter les troues

def TransformationProjective(I, PE, PS):
    h, w = I.shape
    S = np.zeros((h, w))
    matrix = ComputeHomography(PE, PS)
    error_count = 0

    for i in range(h):
        for j in range(w):
            tmp = np.dot(matrix, np.array([[j, i, 1]]).T)
            x, y = round(tmp[0][0]/tmp[2][0]), round(tmp[1][0]/tmp[2][0])
            if 0 < x < h and 0 < y < w:
                # Calcul de l'erreur à retirer
                if S[y][x] != 0:
                    error_count += 1
                S[y][x] = I[i][j]
    return S, error_count
