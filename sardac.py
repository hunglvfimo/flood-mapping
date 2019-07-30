import argparse, sys
import os
import glob

import numpy as np
import cv2

from PIL import Image
import tifffile as tiff

from sklearn.metrics import f1_score, confusion_matrix

parser 	= argparse.ArgumentParser()
parser.add_argument('--val_dir', type=str)
parser.add_argument('--save_dir', type=str)
args 	= parser.parse_args()

def create_binary_array(src_matrix, threshold, small_value, large_value, src_nodata=0, dst_nodata=0):
	"""
	Create binarized image.
	:param src_matrix: Numpy array of source Geotiff.
	:param threshold: Threshold
	:param small_value: Value to be set to pixel below threshold
	:param large_value: Value to be set to pixel above threshold
	:param src_nodata: Value representing nodata of source Geotiff
	:param dst_nodata: Value representing nodata of destination Geotiff
	:return: Numpy array of binarized image
	"""
	bin = np.full(shape=src_matrix.shape, fill_value=large_value, dtype=np.uint8)
	
	bin[src_matrix < threshold] 	= small_value
	bin[src_matrix >= threshold] 	= large_value
	bin[(src_matrix == src_nodata)] = dst_nodata
	
	return bin

def get_threshold(data, nodata=None):
	"""
	Calculate the threshold from the image by Discriminant analysis.
	:param data: Numpy array of source Geotiff.
	:param nodata: Value representing nodata of source Geotiff
	:return: Threshold
	"""
	tmp = data.flatten()
	# If a value of NODATA is specified, exclude the data of that value
	if nodata:
		tmp = tmp[~(tmp==nodata)]

	threshold = cv2.threshold(tmp, 0, 255, cv2.THRESH_OTSU)[0]
	return threshold

def extract_flooded_area(img, threshold=0, filter_size_az=3, filter_size_gr=3):
	img = cv2.blur(img, (filter_size_az, filter_size_gr))

	if threshold == 0:
		threshold = get_threshold(img, nodata=0)

	img_bin = create_binary_array(img, threshold, small_value=2, large_value=1)
	return img_bin, threshold

if __name__ == '__main__':
	y_pred = []
	y_true = []

	for imgpath in glob.glob(os.path.join(args.val_dir, "img", "*.tiff")):
		basename 	= os.path.basename(imgpath).split(".")[0]

		# extract flood areas
		img 	= tiff.imread(imgpath)
		img 	= img[..., 0].astype(np.uint8) # threshold on band HH
		img_bin, threshold 	= extract_flooded_area(img)

		tiff.imsave(os.path.join(args.save_dir, "%s.png" % basename), img_bin)

		# evaluate
		label 	= Image.open(os.path.join(args.val_dir, "label", "%s.png" % basename)).convert('L')
		label	= np.array(label)

		indices_y, indices_x = np.where(label > 0)
		for y, x in zip(indices_y, indices_x):
			if img_bin[y, x] >0 and label[y, x] > 0:
				y_pred.append(img_bin[y, x])
				y_true.append(label[y, x])

	print(f1_score(y_true, y_pred, average='macro'))
	print(f1_score(y_true, y_pred, average='weighted'))
	print(confusion_matrix(y_true, y_pred))