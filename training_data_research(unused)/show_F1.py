import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
import argparse
import math

def getF1(y_true, y_prob, threshold=0.5):
	y_pred = [1 if i >= threshold else 0 for i in y_prob]
	# print(f1_score(y_true, y_pred))
	return f1_score(y_true, y_pred)


def get_averageF1_in_allFold(resultDir_path):
	files = os.listdir(resultDir_path)
	files_file = [f for f in files if os.path.isfile(os.path.join(resultDir_path, f))]
	# print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
	F1_score = np.zeros(len(files_file))
	for i, result_file in enumerate(files_file):
		# print(result_file)
		df = pd.read_csv(os.path.join(resultDir_path, result_file))
		y_true = df["y_test"].tolist()
		y_prob = df["y_pred"].tolist()
		F1_score[i] = getF1(y_true, y_prob)

	print(f"{args.k_list} のFmeasure")
	print(f"mean: {np.mean(F1_score):.3}\nyerr: {np.std(F1_score)/math.sqrt(len(files_file)):.3}")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--research_name", help="", default="new")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	parser.add_argument("--k_list", default="6")
	parser.add_argument("--ratio", type=int, help="正例に対し何倍の負例を作るか", default="1")
	parser.add_argument("--result_root", default="/Users/ylwrvr/卒論/Koga_code/training_data_research/ep2vec_result/new/K562/chromosomal")
	args = parser.parse_args()

	k_mer_set = ["1", "2", "3", "4", "5", "6", "1,2,3,4,5,6", "1,2", "2,3", "4,5", "3,4", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6"]
	for k_list in k_mer_set:
		args.k_list = k_list
		path = os.path.join(args.result_root, f"{args.k_list}_1", "GBRT_4000")
		get_averageF1_in_allFold(path)

	# path = os.path.join(args.result_root, "GBRT_4000")
	# get_averageF1_in_allFold(path)