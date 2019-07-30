import argparse, sys
import os
import glob

import numpy as np
import pandas as pd
import cv2

from PIL import Image
import tifffile as tiff

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

from tqdm import tqdm

parser 	= argparse.ArgumentParser()
parser.add_argument('--train_dir', 	type=str)
parser.add_argument('--val_dir', 	type=str)
parser.add_argument('--test_dir', 	type=str)
parser.add_argument('--save_dir', 	type=str)
args 	= parser.parse_args()

def generate_dataset(dataset_dir):
	data_table = []
	for imgpath in tqdm(glob.glob(os.path.join(dataset_dir, "img", "*.tiff"))):
		basename 	= os.path.basename(imgpath).split(".")[0]

		img 	= tiff.imread(imgpath)
		
		# evaluate
		label 	= Image.open(os.path.join(dataset_dir, "label", "%s.png" % basename)).convert('L')
		label	= np.array(label)

		indices_y, indices_x = np.where(label > 0)
		for x, y in zip(indices_x, indices_y):
			row 	= []
			
			# calculate features
			vh 		= img[y, x, 1]
			vv 		= img[y, x, 2]
			if vv == 0:
				continue
			
			row.append(vh)
			row.append(vv)
			row.append(vh / vv)
			row.append((vv - vh) / (vv + vh))
			row.append(vh / (vv + vh))
			row.append(vv / (vv + vh))
			row.append(label[y, x])

			data_table.append(row)
	data_table = np.array(data_table)
	return data_table

if __name__ == '__main__':
	headers 	= ["vh", "vv", "polarised_ratio_index", "ndpi", "ndhi", "nvvi", "target"]

	if os.path.isfile(os.path.join(args.save_dir, "train.csv")):
		df_train 	= pd.read_csv(os.path.join(args.save_dir, "train.csv"))
		train_table = df_train.values
	else:
		train_table = generate_dataset(args.train_dir)
		df_train	= pd.DataFrame(data=train_table, index=None, columns=headers)
		df_train.to_csv(os.path.join(args.save_dir, "train.csv"), index=False)

	if os.path.isfile(os.path.join(args.save_dir, "val.csv")):
		df_val 		= pd.read_csv(os.path.join(args.save_dir, "val.csv"))
		val_table 	= df_val.values
	else:
		val_table 	= generate_dataset(args.val_dir)
		df_val		= pd.DataFrame(data=val_table, index=None, columns=headers)
		df_val.to_csv(os.path.join(args.save_dir, "val.csv"), index=False)

	if os.path.isfile(os.path.join(args.save_dir, "test.csv")):
		df_test 	= pd.read_csv(os.path.join(args.save_dir, "test.csv"))
		test_table 	= df_test.values
	else:
		test_table 	= generate_dataset(args.test_dir)
		df_test 	= pd.DataFrame(data=test_table, index=None, columns=headers)
		df_test.to_csv(os.path.join(args.save_dir, "test.csv"), index=False)

	X_train, y_train 	= train_table[:, :-1], train_table[:, -1]
	X_val, y_val 		= val_table[:, :-1], val_table[:, -1]
	X_test, y_test 		= test_table[:, :-1], test_table[:, -1]

	scaler 	= StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_val 	= scaler.transform(X_val)
	X_test 	= scaler.transform(X_test)

	# grid search on predefined val set
	X_trainval 	= np.vstack((X_train, X_val))
	y_trainval 	= np.concatenate([y_train, y_val])
	print(X_train.shape, X_val.shape)
	print(y_train.shape, y_val.shape)
	
	val_fold 	= np.zeros(X_trainval.shape[0], dtype=np.uint8)
	val_fold[:X_train.shape[0]] = -1
	ps 			= PredefinedSplit(val_fold)

	rfc 		= RandomForestClassifier(random_state=28)
	param_grid 	= {
					'n_estimators': [200, 500],
					'max_features': ['auto', 'sqrt', 'log2'],
					'max_depth' : [4, 5, 6, 7, 8],
					'criterion' :['gini', 'entropy']
					}
	clf 		= GridSearchCV(estimator=rfc, param_grid=param_grid, cv=ps, n_jobs=4)
	clf.fit(X_trainval, y_trainval)
	print('best params')
	print(clf.best_params_)

	dump(clf, os.path.join(args.save_dir, 'rf.joblib'))

	# evaluate
	y_pred 		= clf.predict(X_test)
	print(f1_score(y_test, y_pred, average='macro'))
	print(f1_score(y_test, y_pred, average='weighted'))