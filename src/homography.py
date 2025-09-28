import numpy as np
from src import main as m
from numba import njit
import matplotlib.pyplot as plt

from src.filter import dilatation


@njit
def compute_homography_matrix(entryPoints, endPoints):
    """
    Compute homography matrix from entryPoints and endPoints for projective_transform.
    At least 4 non-alligned points are required for a valid solution.
    entryPoints and endPoints must contain the same number of points.
    :param entryPoints: Source coordinates in the input image.
    :param endPoints: Target coordinates for the source points.
    :return: Homography matrix used to transform the complete image.
    """
    assert len(entryPoints) == len(endPoints), "compute_homography_matrix() : entryPoints and endPoints must contain the same number of points"
    assert len(entryPoints) >= 4, "compute_homography_matrix() : at least 4 points are required for a solution"

    # Gaussian pivot method used
    # Points are represented with homogeneous coordinates

    A = []
    B = []
    for i in range(len(entryPoints)):
        xe, ye = entryPoints[i][0], entryPoints[i][1]
        xs, ys = endPoints[i][0], endPoints[i][1]
        A.append([xe, ye, 1, 0, 0, 0, -xe*xs, -ye*xs])
        B.append([xs])
        A.append([0, 0, 0, xe, ye, 1, -xe * ys, -ye * ys])
        B.append([ys])
    A = np.array(A)
    B = np.array(B)

    h = np.dot(np.linalg.inv(A), B)
    H = np.array([[h[0][0], h[1][0], h[2][0]],
                  [h[3][0], h[4][0], h[5][0]],
                  [h[6][0], h[7][0], 1]])

    return H


def ErreurNonAffecte(I, C):
    x = C[1][0] - C[0][0]
    y = C[3][0] - C[2][0]
    error = 0
    for i in range(x):
        for j in range(y):
            if I[i+C[0][0],j+C[2][0]] == 0:
                error += 1
    return error


@njit
def interpolation_round(Point, S):
    return round(Point[0]), round(Point[1])


@njit
def projective_transform(I, entryPoints, endPoints, Interpolation : callable = interpolation_round):
    """
    Applies a projective transformation to an image, mapping entryPoints to endPoints.

    The transformation ensures that each coordinate in entryPoints is projected to its corresponding
    coordinate in endPoints. Due to the discrete nature of pixel grids, interpolation is used to handle
    non-integer coordinates, and values are rounded to the nearest pixel.

    :param I: Image
    :param entryPoints: Source coordinates in the input image.
    :param endPoints: Target coordinates for the source points.
    :param Interpolation: Interpolation method used for non-integer coordinates.
    :return: Projected image after transformation.
    """
    h, w = I.shape
    S = np.full((w, h), -1)
    matrix = compute_homography_matrix(entryPoints, endPoints).astype(np.float64)
    error_count = 0
    for i in range(h):
        for j in range(w):
            vec =  np.array([j, i, 1], dtype=np.float64)
            tmp = matrix @ vec
            x, y = tmp[0]/tmp[2], tmp[1]/tmp[2]
            x, y = Interpolation([x, y], S) # Interpolation method
            if 0 < x < w and 0 < y < h:
                # Calcul de l'erreur
                if S[x][y] != -1:
                    error_count += 1
                S[x][y] = I[i][j]
    return S.transpose(), error_count


def get_majority_point(valueList):
    """
    Selects and returns the value from valueList that is closest to the average of all values in the list.
    This function effectively picks the most representative value.
    Used for nearest neighbors function.
    :param valueList: List of values
    :return: Value from the list valueList closest to the average value.
    """
    if len(valueList) == 0:
        return -1
    averageValue = np.mean(valueList)
    distanceDictionary = {abs(valueList[i] - averageValue): i for i in range(len(valueList))}
    vmin = np.min(np.abs(valueList - averageValue))
    return valueList[distanceDictionary[vmin]]


def nearest_neighbor_value(M, P : (int, int), r=1):
    """
    Determines the value at coordinate P in matrix M by aggregating values from its neighbors within a radius r.
    :param M: Matrix
    :param P: Coordinate (x,y) whose value is to be determined.
    :param r: Radius defining the neighborhood around P.
    :return: Computed value for P, based on its neighboring values.
    """
    x, y = P
    valueNeighborsList = []
    for i in range(-r, r+1):
        for j in range(-r, r+1):
            if 0 <= x+i < M.shape[0] and 0 <= y+j < M.shape[1]:
                if  M[x+i][y+j] >= 0:
                    valueNeighborsList.append(M[x+i, y+j])
    return get_majority_point(valueNeighborsList)


def fill_based_on_neighbors(M):
    """
    Fills all occurrences of -1 (unaffected values) in the input matrix using a nearest neighbors algorithm.
    :param M: Matrix
    :return: Filled matrix.
    """
    x, y = M.shape
    Img = M.copy()
    while -1 in M:
        for i in range(x):
            for j in range(y):
                if M[i][j] == -1:
                    Img[i, j] = nearest_neighbor_value(M, (i, j), 1)
        M = Img.copy()
    return Img


def sort_convention(L):
    """
    Sorts a list of points into a standardized order and returns the result.
    The output list follows this order:
    - Index 0: Top-Left
    - Index 1: Top-Right
    - Index 2: Bottom-Right
    - Index 3: Bottom-Left
    :type L: List of points to be sorted, where each point is represented as (x, y).
    """
    assert len(L)==4, "Error : sort_convention require 4 corners points. " + str(len(L)) + " given."
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


def dist_2D(A, B):
    return np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


def merge_points_by_distance(P, d):
    """
    Removes points that are too close to each other based on a minimum distance threshold.
    :param P: List of input points.
    :param d: Minimum required distance between points.
    :return: Filtered list of points, ensuring no two points are closer than d.
    """
    for i in range(len(P)):
        R = []
        for j in range(i+1, len(P)):
            if dist_2D(P[i], P[j]) < d:
                R.append(P[j])
        for r in R:
            P.remove(r)


def barycenter_label_matrix(Label):
    """
    Calculates the barycenter (geometric center) of each labeled island in a connected-component matrix.
    :param Label: Connected-component labeled matrix (output of connected_component_labeling).
    :return: Dictionary mapping each label (key) to the (x, y) coordinates of its barycenter (value).
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


def replace_islands_out_of_range(I, minCount, maxCount, background, replaceValue):
    """
    Islands pixel count that are not in range [minCount, maxCount] are replaced with replaceValue.
    :param I: Image
    :param minCount: minimum pixels
    :param maxCount: maximum pixels
    :param background: Background value that will be ignored (not an island)
    :param replaceValue: New value of these islands
    """
    x, y = I.shape
    LabelsMatrix = m.connected_component_labeling(I, background)
    labels, count = np.unique(LabelsMatrix, return_counts=True)
    LabelCount = dict(zip(labels, count))
    for i in range(x):
        for j in range(y):
            if not(maxCount > LabelCount[LabelsMatrix[i,j]] > minCount):
                I[i,j] = replaceValue


def find_corners_positions(I):
    """
    Find corner position of the image.
    :param I: Matrice image
    :return: Liste de tuple des coordonnées des coins
    """
    VX, VY = m.matrix_2Dgradient(I)
    for i in range(5):
        # Blur the image to extend the intersection
        VX = m.make_gaussian_blur(abs(VX), 5, 10)
        VY = m.make_gaussian_blur(abs(VY), 5, 10)
    m.apply_treshold(VX, np.max(VX)/1.5)
    m.apply_treshold(VY, np.max(VX)/1.5)
    VX = dilatation(VX, 2, 255, 255)
    VY = dilatation(VY, 2, 255, 255)
    C = m.hadamard_product(abs(VX), abs(VY))
    m.print_img(C)
    return barycenter_label_matrix(m.connected_component_labeling(C, 0))


def points_to_matrix(P):
    """
    Converts a list of (x,y) coordinates into a matrix representation.
    :param P: PointList
    :return: np.array
    """
    R = np.zeros((len(P), 2))
    for i in range(len(P)):
        R[i,1] = P[i][0]
        R[i,0] = P[i][1]
    return R


def make_projective_transform(M, treshold = 128, textIslandSize = 2000,
                              imageFormat = np.array([[0, 3508], [2408, 3508], [0, 0], [2408, 0]])):
    """
    Perform an automatic projective transform on a document.
    This function combines algorithms to build the transformation from scratch.
    :param imageFormat: Image format, rectangle
    :param treshold: Initial threshold used to binarize the image and detect edges. Default is 256/2.
    :param textIslandSize: Dark islands with fewer than textIslandSize pixels will be removed.
    :param M: Image matrix.
    :return: Transformed image.
    """

    compressionFactor = 5
    inclinationFactor = .65     # areaEndShape = areaInitialShape * inclinationFactor

    # Automation process
    R = M.copy()
    R = m.compressed_copy(R, compressionFactor)     # Compress the image to speed up the edge dection.
    R = m.make_gaussian_blur(R, 10, 1)
    m.apply_treshold(R, treshold)
    replace_islands_out_of_range(R, textIslandSize//compressionFactor, R.shape[0]*R.shape[1], 255, 255) # Remove text
    cornerList = find_corners_positions(R)

    #if len(cornerList) > 4:
    #    merge_points_by_distance(cornerList, M.shape[0] / 10)

    assert len(cornerList)==4, "transform_projective(). Must have 4 points, got " + str(len(cornerList)) + " instead."
    cornerList = [(x*compressionFactor, y*compressionFactor) for x, y in cornerList] # Adjust the compression factor

    cornerList = sort_convention(cornerList)     # Order points : Top-Left,Top-Right,Bottom-Right,Bottom-Left
    endPoints = imageFormat # A4 Default Document format

    # Determine the optimal end position to minimize the number of unaffected points
    # To achieve this, we maintain the relationship: area_EndPointFigure = area_CornerPointsFigure * multiplier
    # accounting for the effect of inclination.
    areaEndPointsFigure = 2408 * 3508
    X, Y = zip(*cornerList)
    areaCornerPointsFigure = (max(X) - min(X)) * (max(Y) - min(Y))
    endPoints = np.sqrt(areaCornerPointsFigure / areaEndPointsFigure) * endPoints * inclinationFactor
    cornerList = points_to_matrix(cornerList)

    # Scatters preview endpoints and corner points for visualization purposes.
    endPointsX, endPointsY = zip(*endPoints)
    cornerPointsX, cornerPointsY = zip(*cornerList)
    plt.scatter(cornerPointsX, cornerPointsY, color = 'r', label='Starting points (corners)') # Corner Points
    plt.scatter(endPointsX, endPointsY, color='b', label='End points') # End Points
    plt.legend()
    plt.imshow(M, cmap="gray")
    plt.show()

    # Given the start and end points, proceed with the projection calculation.
    TransformedImage, errorCount_overaffectations = projective_transform(M, cornerList, endPoints)
    print("Count of pixels affected by multiple instances (overlap occurrences) : " + str(errorCount_overaffectations))
    return m.cut_img(TransformedImage, (0, 0), (int(endPoints[1, 1]), int(endPoints[1, 0])))