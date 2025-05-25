import os
import main as m
from PIL import Image
import numpy as np


data = []
CharactersNameList = os.listdir('./font/export')
for c in CharactersNameList:
    I = np.array(Image.open("./font/export/" + c, mode="r").convert('L'))
    m.Seuil(I, 128)
    ImgCh, xmin, ymin, xmax, ymax = m.AutoCrop(I, 0)
    data.append((ImgCh, c[0]))

