import pandas as pd
import numpy as np
import os
import io
import subprocess
import tempfile
from glob import glob

import argparse

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import warnings, json, gzip

from torch import Size


from line_notify import line_notify

INF = 9999999999

def get_classifier(args):
	return GradientBoostingClassifier(n_estimators = args.tree, learning_rate = args.alpha, max_depth = args.depth, max_features = 'log2', random_state = args.seed, verbose=1)
	# return svm.SVC(kernel='linear', random_state=0, probability=True)
	# return LogisticRegression(penalty='l2', solver="sag")


def get_weights(y):

	weights_dic = {
		0: 1 / (np.sum(y==0) / len(y)), # 負例重み
		1: 1 / (np.sum(y==1) / len(y)) # 正例重み
	}

	weights_arr = np.zeros(len(y))

	for i in range(len(y)):
		weights_arr[i] = weights_dic[y[i]]

	return weights_arr


def holdout(args, df):

	# print("hold out...")
	_nonpredictors = ["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start","window_end","window_start","window_chrom","window_name","interactions_in_window","active_promoters_in_window"]
	nonpredictors = [f for f in _nonpredictors if f in df.columns]
	# nonpredictors = ['enhancer_chrom', 'enhancer_start', 'enhancer_end', 'promoter_chrom', 'promoter_start', 'promoter_end', 'label', "enhancer_name", "promoter_name"]
	train_chroms=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18"]
	test_chroms=["chr19", "chr20", "chr21", "chr22", "chrX"]

	metrics = {
		"F1": {},
	}

	output_path = os.path.join(args.outdir, args.outname + ".csv")
	# print(f"test chromosome : {test_chroms}")

	train_df = df[df["enhancer_chrom"].isin(train_chroms)]
	# print(train_df.head())
	# valid_df = df[df["enhancer_chrom"].isin(valid_chroms)]
	test_df = df[df["enhancer_chrom"].isin(test_chroms)]
	# print(test_df.head())

	x_train = train_df.drop(columns=nonpredictors).values
	y_train = train_df["label"].values.flatten()
	
	weights = get_weights(y_train)
	classifier = get_classifier(args)

	# print(f"5/{len(x_train)} of x_train")
	# print(f"each has {len(x_train[0])} features")
	# for i in range(5):
	# 	print(x_train[i], f"label={y_train[i]}")

	# print(x_train[0])
	x_train = np.nan_to_num(x_train, nan=0, posinf=0)
	classifier.fit(x_train, y_train, sample_weight=weights) # 学習

	x_test = test_df.drop(columns=nonpredictors).values
	x_test = np.nan_to_num(x_test, nan=0, posinf=0)
	y_test = test_df["label"].values.flatten()
	y_pred = classifier.predict_proba(x_test) # predict
	y_pred = [prob[1] for prob in y_pred] # 正例確率のみを抽出

	result_df = pd.DataFrame(
		{
			"y_test": y_test,
			"y_pred": y_pred
		},
		index=None
	)
	result_df.to_csv(output_path)

	result_df.loc[result_df["y_pred"] > 0.5, "y_pred"] = 1
	result_df.loc[result_df["y_pred"] <= 0.5, "y_pred"] = 0
	print(f"F : {f1_score(y_test, result_df['y_pred'].tolist())}")


def datalist_to_dataframe(data_list):
	row_size = 0
	column_size = 0
	df_list = []
	for data in data_list:
		df_tmp = pd.read_csv(data, sep=",")
		row_size += len(df_tmp)
		# if column_size == 0:
		# 	column_size = len(df_tmp.columns)
		# else:
		# 	assert column_size == len(df_tmp.columns)
		df_list.append(df_tmp)
	df = pd.concat(df_list)
	assert len(df) == row_size
	df = df.fillna(0)
	# print(df.head())
	return df


def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--dataname")
	p.add_argument("--datatype")
	# p.add_argument('-o', "--outdir", required=True, help="Output directory")
	p.add_argument("--outdir", help="Output directory")
	p.add_argument("--tree", type=int, default=4000, help="number of trees")
	p.add_argument("--depth", type=int, default=25, help="number of depth")
	p.add_argument("--alpha", type=float, default=0.001, help="learning rate")
	p.add_argument("--region", default="epw")
	p.add_argument("--outname", help="Output filename")
	p.add_argument('--seed', type=int, default=2020, help="Random seed")

	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()
	np.random.seed(args.seed)

	# print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "condifig_dataset", "*.json"))


	for dataname in ["BENGI"]:
		for datatype in ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", f"maxflow_{INF}"]:
			for region in ["ep", "epw"]:
				for cl in ["GM12878", "HeLa", "K562", "NHEK", "IMR90"]:
					# for config_file in glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_dataset", "*.json")):
					for tree in [100, 500, 1000, 1500]:
						for depth in [10, 15, 20, 25]:
							for alpha in [0.001, 0.0001]:

								args.dataname = dataname
								args.datatype = datatype
								args.region = region
								args.tree = tree
								args.depth = depth
								args.alpha = alpha



								# config = json.load(open(args.config))

								# args.outname = os.path.basename(os.path.splitext(args.config)[0])
								args.outname = cl
								args.outname += f",{args.tree}"
								args.outname += f",{args.depth}"
								args.outname += f",{args.alpha}"


								# data_list = config["datasets"]
								data_list = glob(os.path.join(os.path.dirname(__file__), "data", args.region, "w_feature", dataname, datatype, f"{cl}*.csv"))
								if len(data_list) == 0:
									continue
								print(data_list)
								df = datalist_to_dataframe(data_list)

								args.outdir = f"./output/{args.region}/{args.dataname}_{args.datatype}"
								# args.outdir = config["outdir"]

								if not os.path.exists(args.outdir):
									os.mkdir(args.outdir)

								if os.path.exists(os.path.join(args.outdir, args.outname + ".csv")):
									print(f'{os.path.join(args.outdir, args.outname + ".csv")} has already exsisted!!')
									continue
								

								# holdout(args, df)

								try:
									start_txt = f"開始 args: {args}"
									line_notify(start_txt)
									holdout(args, df)
								except Exception as e:
									line_notify(e)
									text = f"{args.dataname} {args.datatype} {cl} error!!"
									line_notify(text)
								else:
									text = f"TargetFinder tool:\n{args.dataname} {args.datatype} {cl}\ntree={args.tree} depth={args.depth} alpha={args.alpha} finished!!"
									line_notify(text)





