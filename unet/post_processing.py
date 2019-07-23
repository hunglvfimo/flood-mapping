import glob
import os

import numpy as np
import cv2

from scipy.ndimage.morphology import binary_dilation, binary_closing

def post_process():
    kernel = np.ones((3, 3),np.uint8)

    for path in glob.glob(os.path.join("D:\\Workspace\\results\\pisar\\scences", "*.png")):
        basename = os.path.basename(path)

        img = cv2.imread(path, 0)

        img = cv2.dilate(img, kernel, iterations=1)

        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(os.path.join("D:\\Workspace\\results\\pisar\\scences_post_process", basename), img)

if __name__ == '__main__':
    post_process()