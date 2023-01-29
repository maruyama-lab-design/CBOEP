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
import sklearn.metrics as mt

import warnings, json, gzip

INF = 9999999999

def get_classifier(args):
	return GradientBoostingClassifier(n_estimators = args.gbdt_tree, learning_rate = args.gbdt_alpha, max_depth = args.gbdt_depth, max_features ='log2', random_state = 2023, verbose=1)


def get_weights(y):

	weights_dic = {
		0: 1 / (np.sum(y==0) / len(y)),
		1: 1 / (np.sum(y==1) / len(y))
	}

	weights_arr = np.zeros(len(y))

	for i in range(len(y)):
		weights_arr[i] = weights_dic[y[i]]

	return weights_arr


def train(args, df):

	_nonpredictors = ["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start","window_end","window_start","window_chrom","window_name","interactions_in_window","active_promoters_in_window"]
	nonpredictors = [f for f in _nonpredictors if f in df.columns]
	train_chroms = args.train_chroms


	train_df = df[df["enhancer_chrom"].isin(train_chroms)]

	x_train = train_df.drop(columns=nonpredictors).values
	y_train = train_df["label"].values.flatten()
	
	weights = get_weights(y_train)
	classifier = get_classifier(args)

	x_train = np.nan_to_num(x_train, nan=0, posinf=0)
	classifier.fit(x_train, y_train, sample_weight=weights) # train

	return classifier



def test(args, classifier, df):
	_nonpredictors = ["bin","enhancer_chrom","enhancer_distance_to_promoter","enhancer_end","enhancer_name","enhancer_start","label","promoter_chrom","promoter_end","promoter_name","promoter_start","window_end","window_start","window_chrom","window_name","interactions_in_window","active_promoters_in_window"]
	nonpredictors = [f for f in _nonpredictors if f in df.columns]
	test_chroms = args.test_chroms

	metrics = {
		"MCC": -99,
		"balanced accuracy": -99,
		"F1": -99
	}

	output_path = os.path.join(args.outdir, args.outname)
	test_df = df[df["enhancer_chrom"].isin(test_chroms)]

	x_test = test_df.drop(columns=nonpredictors).values
	x_test = np.nan_to_num(x_test, nan=0, posinf=0)
	y_test = test_df["label"].values.flatten()

	y_pred = classifier.predict_proba(x_test) # predict
	y_pred = [prob[1] for prob in y_pred]

	result_df = pd.DataFrame(
		{
			"y_test": y_test,
			"y_pred": y_pred
		},
		index=None
	)
	result_df.to_csv(output_path)

	true = df["y_test"].to_list()
	prob = df["y_pred"].to_list()
	pred =  list(map(round, prob))
	metrics["F1"] = mt.f1_score(true, pred)
	metrics["balanced accuracy"] = mt.balanced_accuracy_score(true, pred)
	metrics["MCC"] = mt.matthews_corrcoef(true, pred)

	print(metrics)


def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--data", default="BENGI")
	p.add_argument("--NIMF_max_d", type=int, default=2500000)
	p.add_argument("--use_window", type=bool, default=True)
	p.add_argument("--train_cell", default="GM12878")
	p.add_argument("--test_cell", default="GM12878")
	p.add_argument("--train_chroms")
	p.add_argument("--test_chroms")
	p.add_argument("--gbdt_tree", type=int, default=1500)
	p.add_argument("--gbdt_depth", type=int, default=10)
	p.add_argument("--gbdt_alpha", type=float, default=0.01)
	p.add_argument("--outdir", default="prediction")
	p.add_argument("--outname", default="result.csv")
	
	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()

	config = json.load(open(os.path.join(os.path.dirname(__file__), "main_opt.json")))
	args.data = config["data"]
	args.NIMF_max_d = config["NIMF_max_d"]
	args.use_window = config["use_window"]
	args.train_cell = config["train_opt"]["train_cell"]
	args.test_cell = config["train_opt"]["test_cell"]
	args.train_chroms = config["train_opt"]["train_chroms"]
	args.test_chroms = config["train_opt"]["test_chroms"]
	args.gbdt_tree = config["model_opt"]["gbdt_tree"]
	args.gbdt_depth = config["model_opt"]["gbdt_depth"]
	args.gbdt_alpha = config["model_opt"]["gbdt_alpha"]

	if args.NIMF_max_d == -1: # original TargetFinder (or BENGI)
		if args.use_window:
			args.outdir = os.path.join(os.path.dirname(__file__), config["outdir"], args.data, f"original", "epw")
			train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"original", "epw", f"{args.train_cell}.csv"))
			test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"original", "epw", f"{args.test_cell}.csv"))
		else:
			args.outdir = os.path.join(os.path.dirname(__file__), config["outdir"], args.data, f"original", "ep")
			train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"original", "ep", f"{args.train_cell}.csv"))
			test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"original", "ep", f"{args.test_cell}.csv"))
	else: # NIMF
		if args.use_window:
			args.outdir = os.path.join(os.path.dirname(__file__), config["outdir"], args.data, f"NIMF_{args.NIMF_max_d}", "epw")
			train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "epw", f"{args.train_cell}.csv"))
			test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "epw", f"{args.test_cell}.csv"))
		else:
			args.outdir = os.path.join(os.path.dirname(__file__), config["outdir"], args.data, f"NIMF_{args.NIMF_max_d}", "ep")
			train_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "ep", f"{args.train_cell}.csv"))
			test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "featured_pair_data", args.data, f"NIMF_{args.NIMF_max_d}", "ep", f"{args.test_cell}.csv"))
	args.outname = f"{args.train_cell}-{args.test_cell},{args.gbdt_tree}-{args.gbdt_depth}-{args.gbdt_alpha}.csv"
	print(args)

	if not os.path.exists(args.outdir):
		os.makedirs(args.outdir, exist_ok=True)

	# match feature order in train and test
	train_columns = set(list(train_df.columns))
	test_columns = set(list(test_df.columns))
	print(f"train columns: {len(train_columns)}")
	print(f"test columns: {len(test_columns)}")
	union_columns = train_columns | test_columns
	print(f"union columns: {len(union_columns)}")
	train_df[list(union_columns - train_columns)] = 0
	test_df[list(union_columns - test_columns)] = 0
	train_df = train_df[list(test_df.columns)]

	assert list(train_df.columns) == list(test_df.columns), "feature order error"

	classifier = train(args, train_df) # train
	test(args, classifier, test_df) # test






