from PIL import Image
import numpy as np
import pandas as pd
import os
import sys
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt


def load_img(img_id, new_size, aug=True):
    trg = pd.read_csv(os.path.join(sys.path[0], "trg.csv"))
    print(trg.head())

    img = Image.open(os.path.join(sys.path[0], "crops/", "img_" + str(img_id + 1) + ".jpg"))
    imw, imh = img.size
    img_arr = np.array(img, dtype=np.float32) / 255

    X1 = int(trg.iloc[img_id].X1 * imw)
    X2 = int(trg.iloc[img_id].Xo * imw)
    Y1 = imh
    Y2 = int(trg.iloc[img_id].Yo * imh)
    W = int(trg.iloc[img_id].W * imw)
    kmax_crop = 0.3
    maxmax_crops = (int(imw * kmax_crop), int(imh * kmax_crop))
    max_crops = (min(max(min(X1, X2) - int(W / 2), 0), maxmax_crops[1]),
                 min(max(imw - max(X1, X2) - int(W / 2), 0), maxmax_crops[0]),
                 min(max(imh - Y2, 0), maxmax_crops[0]))

    aug_list = []
    if aug:
        aug_list += [iaa.Crop(px=((0, max_crops[2]),
                                  (0, max_crops[1]),
                                  (0, 0),
                                  (0, max_crops[0]))),
                     iaa.Fliplr(p=0.5)]
    aug_list += [iaa.Resize(new_size)]

    seq = iaa.Sequential(aug_list)
    seq_det = seq.to_deterministic()

    augmented_kp = []
    kp = []

    kp.append(ia.Keypoint(x=int(X1 - W / 2), y=Y1))
    kp.append(ia.Keypoint(x=int(X1 + W / 2), y=Y1))
    kp.append(ia.Keypoint(x=X2, y=Y2))
    augmented_kp.append(ia.KeypointsOnImage(kp, shape=img_arr.shape))

    img_arr = seq_det.augment_image(img_arr)
    augmented_kp = seq_det.augment_keypoints(augmented_kp)[0].keypoints
    return np.clip(img_arr, 0, 1), augmented_kp


def show_img(img, p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x0 = int(np.mean([x1, x2]))

    plt.plot([x0, x3], [y1, y3], "--", alpha=0.6)
    plt.plot([x1, x3], [y1, y3], c='blue', linewidth=3, alpha=0.6)
    plt.plot([x2, x3], [y2, y3], c='blue', linewidth=3, alpha=0.6)
    plt.imshow(img)
    plt.show()
