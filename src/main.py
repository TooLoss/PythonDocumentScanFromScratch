import numpy as np
import matplotlib.pyplot as plt
from numba import njit # Précompile le code, execution plus rapide
from numba.typed import Dict


@njit
def hadamard_product(A, B):
    """
    Computes the Hadamard product of two matrices.
    The Hadamard product multiplies corresponding elements of A and B.
    Both matrices must have identical dimensions.
    :param A: Matrix
    :param B: Matrix
    :return: Matrix A (.) B
    """
    x, y = A.shape
    R = np.zeros((x, y), A.dtype)
    for i in range(x):
        for j in range(y):
            R[i,j] = A[i,j] * B[i,j]
    return R


@njit
def convolve_at(A, i, j, C):
    """
    Make convolution operation on A at the coordinate (i,j) using kernel C.
    :param A: Matrix
    :param i: Coordinate
    :param j: Coordinate
    :param C: Convolution Matrix
    :return: Result of convolution operation
    """
    mx, my = A.shape
    x, y = C.shape
    cx = (x-1)//2
    cy = (y-1)//2
    S = 0
    for i in range(x):
        for j in range(y):
            if (0 <= (i - cx) + i < mx) and (0 <= (j - cy) + j < my):
                S += A[i - cx + i, j - cy + j] * C[i, j]
    return S


@njit
def convolve_matrix(A, C):
    """
    Make complete convolution operation on A (for each coordinates) using kernel C.
    :param A: Matrix
    :param C: Convolution Matrix
    :return: Convolved Matrix
    """
    x, y = A.shape
    R = np.zeros((x, y), A.dtype)
    for i in range(x):
        for j in range(y):
            R[i, j] = convolve_at(A, i, j, C)
    return R


def gaussian_filter(n, s):
    """
    Compute gaussian filter centered.
    :param n: size matrix
    :param s: sigma value (std deviation)
    :return: Matrix size n Gaussian Filter
    """
    G = np.zeros((n, n))
    M = (n+1)//2
    for i in range(n):
        for j in range(n):
            G[i, j] = (1/2*np.pi*(s**2)) * \
                np.exp(-((i-(M-1))**2 + (j-(M-1))**2)/(2 * s**2))
    return G/np.sum(G) # Normalise pour conserver la même luminosité 


def apply_gaussian_blur(A, n, s):
    """
    Apply gaussian blur on A.
    :param A: Image
    :param n: filter size
    :param s: sigma value
    :return: Blured image
    """
    return convolve_matrix(A, gaussian_filter(n, s))


def matrix_2Dgradient(M, CX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), CY=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).transpose()):
    """
    Canny's edge detection algorithm.
    :param M: Iamge
    :param CX: X gradient filter (Sobel by default)
    :param CY: Y gradient filter (Sobel by default)
    :return: X gradient Matrix, Y gradient Matrix
    """
    MX = convolve_matrix(M, CX)
    MY = convolve_matrix(M, CY)
    return MX, MY


def compressed_copy(M, f):
    """
    Make a copy and compress image M by f
    :param M: Image
    :param f: Compression factor
    :return: Compressed image
    """
    if f == 1:
        return M
    x, y = M.shape
    I = np.zeros((x//f, y//f))
    for i in range(x//f):
        for j in range(y//f):
             I[i, j] = M[i*f, j*f]
    return I


def print_img(M, scale=1):
    """
    Print image matrix without axis.
    :param M: Image
    :param scale: Exit size multiplier.
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8.3*scale, 11.7*scale, True) # A4
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(M, cmap='gray', interpolation='none')


def print_img_in_range(M, vmin=0, vmax=255):
    """
    Print image matrix without axis in range [vmin,vmax]
    :param M: Image
    :param vmax: max value
    :param vmin: min value
    """
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8.3, 11.7, True) # A4
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(M, cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)


def save_img_as_png(M, filename : str):
    """
    Save image as PNG
    :param M: Image
    :param filename: Absolute destination location
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


def treshold(M, s):
    x, y = M.shape
    for i in range(x):
        for j in range(y):
            if M[i,j] < s:
                M[i,j] = 0
            else:
                M[i,j] = 255
    return M


@njit
def get_surrounding(I, position : (int,int), radius):
    """
    Renvoie la liste des valeurs des pixels autours de la position pos
    :param I: Image
    :param position: tuple (x,y) coordinates
    :param radius: radius considéré
    """
    L = []
    x, y = position
    maxx, maxy = I.shape
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if 0 <= i < maxx and 0 <= j < maxy:
                L.append(I[i,j])
    return L


@njit
def cut_img(M, A : (int, int), B : (int, int)):
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


def connected_component_labeling(I, backgroundValue = 0):
    """
    Detect islands in the image.
    Islands are separated with background. This is why backgroundValue is ignored.
    :param backgroundValue: Background color
    :param I: Iamge
    :return: Matrix with labels that tell wich island is in (x,y).
    """
    x, y = np.shape(I)
    Label = np.zeros((x, y), dtype=int)
    tag = 1
    Redef = {}
    for i in range(1, x-1):
        for j in range(1, y-1):

            LVoisin = [Label[i-1,j], Label[i,j-1]]
            LVoisin = [i for i in LVoisin if i != 0]

            if I[i,j] != backgroundValue:
                if len(LVoisin) == 0: # if it is a new island
                    Label[i,j] = tag
                    tag += 1
                else: # if the island is connected to another one, then flag their numbers
                    min_tag = min(LVoisin)
                    Label[i, j] = min_tag
                    for n in LVoisin:
                        if n != min_tag:
                            Redef[n] = min_tag
    # merged island numbers that touch each others
    for i in range(x):
        for j in range(y):
            if Label[i, j] != 0:
                label = Label[i, j]
                # get the minimum value of all the hitted islands to merge them
                while label in Redef:
                    label = Redef[label]
                Label[i, j] = label

    return Label


@njit
def auto_crop(I, value=0):
    """
    Resize the image to contain only the value selected
    :param I: Image
    :param value: value the autocrop will isolate
    :return:
    """
    x, y = I.shape
    xmin, ymin, = x-1, y-1
    xmax, ymax = 0, 0
    for i in range(x):
        for j in range(y):
            if I[i, j] == value:
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