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


def add_value_label(x_list,y_list):
	for i in range(1, len(x_list)+1):
		plt.text(i, y_list[i-1], y_list[i-1], size='x-small')


def getF1(y_true, y_prob, threshold=0.5):
	y_pred = [1 if i >= threshold else 0 for i in y_prob]
	# print(f1_score(y_true, y_pred))
	return f1_score(y_true, y_pred)


def get_averageF1_in_allFold(resultDir_path):
	files = os.listdir(resultDir_path)
	print(f"{files} の F-measure を計算")
	files_file = [f for f in files if os.path.isfile(os.path.join(resultDir_path, f))]
	# print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
	F1_score = np.zeros(len(files_file))
	for i, result_file in enumerate(files_file):
		# print(result_file)
		df = pd.read_csv(os.path.join(resultDir_path, result_file))
		y_true = df["y_test"].tolist()
		y_prob = df["y_pred"].tolist()
		F1_score[i] = getF1(y_true, y_prob)

	print(f"{round(np.mean(F1_score), 3)} ± {round(np.std(F1_score)/math.sqrt(len(files_file)), 3)}")
	return {"mean": np.mean(F1_score), "yerr": np.std(F1_score)/math.sqrt(len(files_file))}


def get_F1_Dict(datasetName, cell_line, classifier, k_mer_set_list):
	result_dict = {}
	for k_mer_set in k_mer_set_list:
		resultDir = os.path.join(os.path.dirname(__file__), ".", "ep2vec_result", datasetName, cell_line, "chromosomal", f"×1", f"{k_mer_set}_1", classifier)
		result_dict[k_mer_set] = get_averageF1_in_allFold(resultDir)
	
	return result_dict


def make_F1_barGraph_by_k_mer_set(datasetName, cell_line, classifier, k_mer_set_list):
	result_dict = get_F1_Dict(datasetName, cell_line, classifier, k_mer_set_list)
	labels = list(result_dict.keys())
	left = list(range(1, len(labels)+1))
	f1 = [result_dict[key]["mean"] for key in labels]
	yerr = [result_dict[key]["yerr"] for key in labels]
	values = [round(v, 3) for v in f1]

	fig, ax = plt.subplots()
	ax.set_ylim([0, 1])
	
	plt.bar(left, f1, width=0.2, color='red', tick_label=labels, yerr=yerr)
	add_value_label(left,values)
	plt.ylabel('F-measure')
	plt.xlabel('k-mer set')
	plt.xticks(rotation=45)
	plt.tick_params(labelsize=8)
	plt.savefig(f"{cell_line}_by_k_mer_set.png",dpi=130,bbox_inches = 'tight', pad_inches = 0)
	fig.set_figheight(8)
	fig.set_figwidth(15)
	plt.show()





if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--research_name", help="", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")

	parser.add_argument("--ratio", type=int, help="正例に対し何倍の負例を作るか", default="1")
	args = parser.parse_args()

	# cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
	# research_list = ["new"]
	# ratio_list = [1, 2, 3, 4, 5]

	make_F1_barGraph_by_k_mer_set("new", "GM12878", "GBRT_4000", ["1", "2", "3", "4", "5", "6", "7", "8", "1,2", "2,3", "3,4", "4,5", "5,6", "1,2,3", "2,3,4", "3,4,5", "4,5,6", "1,2,3,4,5,6", "1,2,3,4,5,6,7,8"])

	# for researchName in research_list:
	# 	for cell_line in cell_line_list:
	# 		for ratio in ratio_list:
	# 			args.ratio = ratio
	# 			args.research_name = researchName
	# 			args.cell_line = cell_line
	# 			make_PosNeg_figure(args)

	# for cell_line in cell_line_list:
	# 	make_F1_barGraph(["TargetFinder", "ep2vec", "new"], cell_line, ["GBRT_100", "GBRT_1000", "GBRT_4000", "KNN_5", "KNN_10", "KNN_15"])
	# make_precision_recall_curve(["new", "TargetFinder", "ep2vec"], ["GBRT_100", "GBRT_1000", "GBRT_4000"])