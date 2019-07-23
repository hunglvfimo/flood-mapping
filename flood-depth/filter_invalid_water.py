import numpy as np

import os
import glob

import cv2
import tifffile as tiff

from tqdm import tqdm

from params import *

input_dir 	= "D:\\Workspace\\results\\pisar\\scences\\water_mask_dilated"
save_dir 	= "D:\\Workspace\\results\\pisar\\scences\\water_mask_valid"

for filepath in tqdm(glob.glob(os.path.join(input_dir, "*.png"))):
	basename 	= os.path.basename(filepath).split(".")[0]

	# read water mask detected from UNet
	water_mask 	= cv2.imread(filepath, 0)

	# read DEM mask
	dem 		= tiff.imread(os.path.join(dem_dir, "%s.tif" % basename))
	slope 		= tiff.imread(os.path.join(slope_dir, "%s.tif" % basename))

	# remove water area where DEM = -9999 (aka ocean area)
	water_mask[dem[...,  0] == NODATA_VAL] = 0
	# remove water area where slope > 5 (aka hill, tree, ...)
	water_mask[slope[..., 0] > SLOPE_THRES] = 0

	cv2.imwrite(os.path.join(save_dir, "%s.png" % basename), water_mask)