
import numpy as np
import matplotlib.pyplot as plt





from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt


def filenames2df(filenames):
	unused_columns = [str(i) for i in range(9)]
	outdf = ""
	for i, filename in enumerate(filenames):
		if i == 0:
			outdf = pd.read_table(filename, header=None, names=["label"]+unused_columns)
		else:
			outdf = pd.concat([outdf, pd.read_table(filename, header=None, names=["label"]+unused_columns)])

	return outdf


def make_olddata_table():
	cell_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]
	dataset_list = ["BG"]
	NIMF_list = ["", "2.5M"]

	table_list = []
	for cell in cell_list:
		for dataset in dataset_list:
			for NIMF in NIMF_list:
				mid_dir = "original_old"
				if NIMF != "":
					mid_dir = "maxflow_old"
				filenames = glob.glob(os.path.join(os.path.dirname(__file__), "data", mid_dir, f"{cell}*.tsv"))
				df = filenames2df(filenames)
				positive = len(df[df["label"] == 1])
				negative = len(df[df["label"] == 0])
				neg_pos = negative / positive
				table_list.append([cell, dataset, NIMF, positive, negative, neg_pos])
	df = pd.DataFrame(data=table_list, columns=["cell", "dataset", "NIMF", "positive", "negative", "neg/pos"])
	df.to_csv('data.csv', index=False)
	with pd.ExcelWriter('data.xlsx', mode="a", if_sheet_exists="replace") as writer:
		df.to_excel(writer, sheet_name='BENGI_old')


def make_data_table():
	cell_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	dataset_list = ["BG"]
	NIMF_list = ["", "2.5M", "3.0M", "3.5M", "4.0M", "4.5M", "5.0M"]

	table_list = []
	for cell in cell_list:
		for dataset in dataset_list:
			for NIMF in NIMF_list:
				mid_dir = "original"
				if NIMF != "":
					mid_dir = "maxflow_" + str(int(NIMF[0]+NIMF[2]) * 100000)
				if not os.path.exists(os.path.join(os.path.dirname(__file__), "data", mid_dir, f"{cell}.tsv")):
					continue
				filename = os.path.join(os.path.dirname(__file__), "data", mid_dir, f"{cell}.tsv")
				unused_columns = [str(i) for i in range(9)]
				df = pd.read_table(filename, header=None, names=["label"]+unused_columns)
				positive = len(df[df["label"] == 1])
				negative = len(df[df["label"] == 0])
				neg_pos = negative / positive
				table_list.append([cell, dataset, NIMF, positive, negative, neg_pos])
	df = pd.DataFrame(data=table_list, columns=["cell", "dataset", "NIMF", "positive", "negative", "neg/pos"])
	df.to_csv('data.csv', index=False)
	with pd.ExcelWriter('data.xlsx') as writer:
		df.to_excel(writer, sheet_name='BENGI')




def make_result_table():
	dataset_list = ["BG"]
	train_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	test_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	datatype_list = ["org", "mf"]
	table_list = []

	for dataset in dataset_list:
		for train in train_list:
			for test in test_list:
				for datatype in datatype_list:
					train_test = train + "-" + test
					NIMF = ""
					if datatype == "mf":
						NIMF = "2.5M"
					filenames = glob.glob(os.path.join(os.path.dirname(__file__), f"{dataset}_{datatype}(cl_wise)_{train}_noScheduler(lr=0.0001)_noMSE", f"*{test}_prediction.txt"))
					if filenames == []:
						continue
					filename = filenames[0]
					df = pd.read_table(filename, skiprows=1, names=["y_test", "y_pred"])
					true = df["y_test"].to_list()
					prob = df["y_pred"].to_list()

					pred =  list(map(round, prob))
					true_prime = [int(i == 0) for i in true]
					pred_prime = [int(i == 0) for i in pred]
					balanced_accuracy = mt.balanced_accuracy_score(true, pred)
					MCC = mt.matthews_corrcoef(true, pred)

					table_list.append([train_test, dataset, NIMF, MCC, balanced_accuracy])
					
	df = pd.DataFrame(data=table_list, columns=["train-test", "dataset", "NIMF", "MCC", "bl-acc"])
	df.to_csv('predictor_result.csv', index=False)
	with pd.ExcelWriter('predictor_result.xlsx') as writer:
		df.to_excel(writer, sheet_name='TransEPI')





make_olddata_table()