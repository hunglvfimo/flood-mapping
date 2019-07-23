import numpy as np

import os
import glob

import tifffile as tiff

from osgeo import gdal

from tqdm import tqdm

from params import *


geo_dir = "D:\\Workspace\\data\\raw\\pi-sar2\\20110312\\tiff_all"

wsl_dir = "D:\\Workspace\\results\\pisar\\scences\\wsl"

dem_dir = "D:\\Workspace\\results\\pisar\\scences\\dem"
slope_dir = "D:\\Workspace\\results\\pisar\\scences\\slope"

save_dir = "D:\\Workspace\\results\\pisar\\scences\\flood_depth"

pbar = tqdm(glob.glob(os.path.join(wsl_dir, "*.tif")))
for filepath in pbar:
	basename 	= os.path.basename(filepath).split(".")[0]

	# reading geo-information from its correspondence image
	geo_img 	= gdal.Open(os.path.join(geo_dir, "%s_sc.tif" % basename))
	arr 		= geo_img.ReadAsArray()
	trans 		= geo_img.GetGeoTransform()
	proj 		= geo_img.GetProjection()

	# read water mask detected from UNet
	wsl_img 	= tiff.imread(filepath)

	# read DEM mask
	dem 		= tiff.imread(os.path.join(dem_dir, "%s.tif" % basename))
	dem[dem == NODATA_VAL] = 0.0 # remove DEM with no_data value

	slope 		= tiff.imread(os.path.join(slope_dir, "%s.tif" % basename))
	slope[slope == NODATA_VAL] = 0.0 # remove slope with no_data value	
	
	# calculate flood depth
	flood_depth_img = wsl_img - dem

	# filter out area with flood depth < 0 caused by error
	flood_depth_img[flood_depth_img < 0] = 0

	# filter out flooded area with high slope (5 degree)
	flood_depth_img[slope > 5] = 0

	outdriver	= gdal.GetDriverByName("GTiff")
	outdata   	= outdriver.Create(os.path.join(save_dir, "%s.tif" % basename), flood_depth_img.shape[1], flood_depth_img.shape[0], 1, gdal.GDT_Float32)
	outdata.GetRasterBand(1).WriteArray(flood_depth_img[..., 0])
	outdata.GetRasterBand(1).SetNoDataValue(0)
	outdata.SetGeoTransform(trans)
	outdata.SetProjection(proj)

	
	