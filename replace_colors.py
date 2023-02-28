import cv2
import glob

import matplotlib.pyplot as plt
import numpy as np
import paths


# Load hyperspectral mask
images = glob.glob(paths.folder_path + 'raw*15040*rf' + paths.class_mask_extension)
for image in images:
    print(image)
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    h = img.shape[0]
    w = img.shape[1]
    id_image = np.zeros(shape=(h, w))
    color_dict = {(0, 0, 0): 0}

    for y in range(0, h):
        for x in range(0, w):
            color = (int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2]))
            if color not in color_dict:
                color_dict[color] = len(color_dict)

    print(color_dict)
    # Color swap
    swap = { (64, 255, 0): (0, 255, 255) }
    for key in swap:
        color_dict[swap[key]] = color_dict[key]
        del color_dict[key]

    for y in range(0, h):
        for x in range(0, w):
            color = (int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2]))
            if color not in color_dict:
                color = swap[color]
                img[y, x, 0] = color[0]
                img[y, x, 1] = color[1]
                img[y, x, 2] = color[2]

    print(image.split(paths.class_mask_extension)[0] + '_2' + paths.class_mask_extension)
    cv2.imwrite(image.split(paths.class_mask_extension)[0] + '_2' + paths.class_mask_extension, img)
