import numpy as np

import os
import glob

import cv2
import geoio
import tifffile as tiff

from tqdm import tqdm

flood_map_dir = "D:\\Workspace\\results\\pisar\\scences\\water_mask"
geo_dir = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\tiff_all"
slope_path = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\slope.tif"
save_dir = "D:\\Workspace\\results\\pisar\\scences\\slope"

slope_img = geoio.GeoImage(slope_path)
slope_data = slope_img.get_data() # (bands, rows, cols)

pbar = tqdm(glob.glob(os.path.join(flood_map_dir, "*.png")))
for filepath in pbar:
	basename = os.path.basename(filepath).split(".")[0]

	flood_img = cv2.imread(filepath, 0)

	flood_slope_img = np.zeros(flood_img.shape + (1,))
	
	geo_path = os.path.join(geo_dir, "%s_sc.tif" % basename)
	geo_img = geoio.GeoImage(geo_path)

	y = np.arange(flood_img.shape[0])
	x = np.arange(flood_img.shape[1])

	yx = [[i, j] for i in y for j in x]
	yx = np.array(yx)

	geo_x, geo_y = geo_img.raster_to_proj(yx[:, 1], yx[:, 0])
	
	slope_x, slope_y = slope_img.proj_to_raster(geo_x, geo_y)
	slope_x = [int(i) for i in slope_x]
	slope_y = [int(i) for i in slope_y]
	
	flood_slope_img[yx[:, 0], yx[:, 1], 0] = slope_data[0, slope_y, slope_x]

	tiff.imsave(os.path.join(save_dir, "%s.tif" % basename), flood_slope_img, planarconfig='contig')