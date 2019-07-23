import numpy as np

import os
import glob

import cv2 as cv
import geoio
import tifffile as tiff

from tqdm import tqdm

from params import *

water_dir 		= "D:\\Workspace\\results\\pisar\\scences\\water_mask_valid"
scattering_dir 	= "D:\\Workspace\\results\\pisar\\scences\\four_component"
save_dir 		= "D:\\Workspace\\results\\pisar\\scences\\objects"

for filepath in glob.glob(os.path.join(scattering_dir, "*.tif")):
	basename		= os.path.basename(filepath).split(".")[0]
	
	# read water mask detected from UNet
	water_mask 		= cv.imread(os.path.join(water_dir, "%s.png" % basename), 0)
	water_mask[water_mask > 0] = 1

	# read double scattering power
	scattering 		= tiff.imread(filepath)
	pd 				= scattering[..., 1]
	pd 				= pd * water_mask

	# threshold
	output 				= np.zeros(water_mask.shape, dtype=np.uint8)
	output[pd >= 205] 	= 255

	tiff.imsave(os.path.join(save_dir, "%s.tif" % basename), output, planarconfig='contig')


