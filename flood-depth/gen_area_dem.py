import numpy as np

import os
import glob

import cv2
import geoio
import tifffile as tiff

flood_map_dir = "D:\\Workspace\\results\\pisar\\scences\\flood_mask_post"
geo_dir = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\tiff_all"
dem_path = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\dem.tif"
save_dir = "D:\\Workspace\\results\\pisar\\scences\\dem"

dem_img = geoio.GeoImage(dem_path)
dem_data = dem_img.get_data() # (bands, rows, cols)

for filepath in glob.glob(os.path.join(flood_map_dir, "*.png")):
	basename = os.path.basename(filepath).split(".")[0]

	flood_img = cv2.imread(filepath, 0)

	flood_dem_img = np.zeros(flood_img.shape + (1,))
	
	geo_path = os.path.join(geo_dir, "%s_sc.tif" % basename)
	geo_img = geoio.GeoImage(geo_path)

	y = np.arange(flood_img.shape[0])
	x = np.arange(flood_img.shape[1])

	yx = [[i, j] for i in y for j in x]
	yx = np.array(yx)

	geo_x, geo_y = geo_img.raster_to_proj(yx[:, 1], yx[:, 0])
	
	dem_x, dem_y = dem_img.proj_to_raster(geo_x, geo_y)
	dem_x = [int(i) for i in dem_x]
	dem_y = [int(i) for i in dem_y]
	
	flood_dem_img[yx[:, 0], yx[:, 1], 0] = dem_data[0, dem_y, dem_x]

	tiff.imsave(os.path.join(save_dir, "%s.tif" % basename), flood_dem_img, planarconfig='contig')