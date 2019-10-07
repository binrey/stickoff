import numpy as np
import pandas as pd
from utils import *


img_size = [224, 224]
imw = img_size[0]
imh = img_size[1]
n = 2
img, kp = load_img(n, img_size, True)

show_img(img,
         [min(kp[0].x, kp[1].x), kp[0].y],
         [max(kp[0].x, kp[1].x), kp[1].y],
         [kp[2].x, kp[2].y])

