import numpy as np

import os
import glob

import cv2 as cv
import geoio
import tifffile as tiff

from tqdm import tqdm

from params import *

input_dir = "D:\\Workspace\\results\\pisar\\scences\\water_mask_valid"
save_dir = "D:\\Workspace\\results\\pisar\\scences\\wsl"

def outliner_removal(data):
	""" Remove outliner with assumtion that data follow Normal Distribution
		Outliner is define by: < mean - 3 * std or > mean + 3 * std
		Input:
			data: numpy 1-d array
	"""
	mean 	= np.mean(data)
	std 	= np.std(data)
	data 	= data[data < mean + 3.0 * std]
	data 	= data[data > mean - 3.0 * std]
	return data

for filepath in glob.glob(os.path.join(input_dir, "*.png")):
	basename		= os.path.basename(filepath).split(".")[0]
	# read water mask detected from UNet
	water_mask 		= cv.imread(filepath, 0)
	# read DEM data
	dem 			= tiff.imread(os.path.join(dem_dir, "%s.tif" % basename))
	# read slope data
	slope 			= tiff.imread(os.path.join(slope_dir, "%s.tif" % basename))
	# initialize water surface level map
	water_level_img = np.zeros(water_mask.shape + (1, ))

	# use for drawing contour
	cnt_mask_tmp 	= np.empty(water_mask.shape, dtype=np.uint8)
	cnt_outline_tmp = np.empty(water_mask.shape, dtype=np.uint8)

	# finding contours (aka flooded area)
	contours, _ 	= cv.findContours(water_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	pbar = tqdm(contours)
	for i, cnt in enumerate(pbar):
		pbar.set_description(basename)

		# ignore if flooded area is too small
		if cv.contourArea(cnt) < FLOOD_SIZE_THRES:
			continue

		# reset temp mask
		cnt_mask_tmp.fill(0)
		cnt_outline_tmp.fill(0)

		# finding pixels belong to outline of the contours
		# by drawing contour into temp image without filling it
		cv.drawContours(cnt_mask_tmp, [cnt], 0, 1, -1)
		cv.drawContours(cnt_outline_tmp, [cnt], 0, 1, 1)

		# finding flood area boundary pixel
		dem_bound 		= dem[:, :, 0] * cnt_outline_tmp
		# filter out outliner
		dem_bound 		= outliner_removal(dem_bound)
		# calculating water surface level according to flooded boundary
		mean_dem_bound 	= np.mean(dem_bound)

		# draw to final water level map
		(water_level_img[:, :, 0])[cnt_mask_tmp == 1] = mean_dem_bound
	
	tiff.imsave(os.path.join(save_dir, "%s.tif" % basename), water_level_img, planarconfig='contig')
