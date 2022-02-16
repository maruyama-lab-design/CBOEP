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


def set_fontsize(axis_Max):
	x1, y1, x2, y2 = 100, 8, 625, 3
	a = (y2 - y1) / (x2 - x1)
	x = axis_Max * axis_Max
	x = max(x, 100)
	x = min(x, 625)
	y = a * (x - x1) + y1
	return y

def make_heatMap(args, chrom, VV, x_Max, y_Max, regionType, output_path):
	axis_Max = max(x_Max, y_Max)
	fontsize = set_fontsize(axis_Max)

	data = VV[:axis_Max+1, :axis_Max+1] # 二次元配列です

	mask = np.zeros_like(data)
	mask[np.where(data==0)] = True    

	plt.figure(figsize=(21, 14))
	fig, ax = plt.subplots()
	sns.heatmap(data, annot=True, square=True, annot_kws={"fontsize":fontsize}, fmt="d", cmap="Blues", linewidths=.1, linecolor='black', mask=mask, cbar = False)
	ax.invert_yaxis()
	# ax.set_title(f"{args.cell_line} pos-neg cnt by each {regionType}")
	ax.set_xlabel("positive example count")
	ax.set_ylabel("negative example count")
	plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0.01)
	plt.close('all')

	
def make_PosNeg_figure(args):
	output_dir = os.path.join(os.path.dirname(__file__), "figure", args.research_name, args.cell_line)
	os.system(f"mkdir -p {output_dir}")

	data_path = os.path.join(os.path.dirname(__file__), "training_data", args.research_name, f"{args.cell_line}_train.csv")
	df = pd.read_csv(data_path)

	for regionType in ["enhancer", "promoter"]:
		posCnt_Max = 0
		negCnt_Max = 0
		PosNeg_cnt = np.zeros((30, 30), dtype="int64") # 大きめに用意
		columnName = regionType + "_name"

		for chrom, subdf in df.groupby("enhancer_chrom"):
			posCnt_Max_by_chrom = 0
			negCnt_Max_by_chrom = 0
			PosNeg_cnt_by_chrom = np.zeros((30, 30), dtype="int64") # 大きめに用意
			for regionName, subsubdf in subdf.groupby(columnName):
				posCnt = len(subsubdf[subsubdf["label"] == 1])
				negCnt = len(subsubdf[subsubdf["label"] == 0])

				PosNeg_cnt_by_chrom[negCnt][posCnt] += 1

				posCnt_Max_by_chrom = max(posCnt_Max_by_chrom, posCnt)
				negCnt_Max_by_chrom = max(negCnt_Max_by_chrom, negCnt)

			output_path = os.path.join(output_dir, f"{chrom}_{regionType}.png")
			make_heatMap(args, chrom, PosNeg_cnt_by_chrom, posCnt_Max_by_chrom, negCnt_Max_by_chrom, regionType, output_path)
			
			PosNeg_cnt += PosNeg_cnt_by_chrom
			posCnt_Max = max(posCnt_Max, posCnt_Max_by_chrom)
			negCnt_Max = max(negCnt_Max, negCnt_Max_by_chrom)

		output_path = os.path.join(output_dir, f"chrAll_{regionType}.png")
		make_heatMap(args, chrom, PosNeg_cnt, posCnt_Max, negCnt_Max, regionType, output_path)


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

	return {"mean": np.mean(F1_score), "yerr": np.std(F1_score)/math.sqrt(len(files_file))}


def get_F1_Dict(datasetNames, cell_line, classifiers):
	result_dict = {}
	for datasetName in datasetNames:
		result_dict[datasetName] = {"random": {}, "chromosomal": {}}
		for classifier in classifiers:
			resultDir = os.path.join(os.path.dirname(__file__), "ep2vec_result", datasetName, cell_line, "chromosomal", classifier)
			result_dict[datasetName]["chromosomal"][classifier] = get_averageF1_in_allFold(resultDir)
			resultDir = os.path.join(os.path.dirname(__file__), "ep2vec_result", datasetName, cell_line, "random", classifier)
			result_dict[datasetName]["random"][classifier] = get_averageF1_in_allFold(resultDir)
	return result_dict


def make_F1_barGraph_by_each_Classifier(datasetNames, cell_line, classifiers):
	result_dict = get_F1_Dict(datasetNames, cell_line, classifiers)
	for dataset in datasetNames: # 調査するdatasetの数だけグラフを作成する．
		fig, ax = plt.subplots()
		ax.set_ylim([0, 1])
		# ax.set_title(f"pseudo ep2vec in {dataset} dataset's K562")
		bar_width = 0.10
		alpha = 0.8
		colorList = ["blue", "red", "green", "pink", "gold", "purple", "green"]
		for i, classifier in enumerate(classifiers):
			chromosomal_result_mean = result_dict[dataset]["chromosomal"][classifier]["mean"]
			random_result_mean = result_dict[dataset]["random"][classifier]["mean"]
			chromosomal_result_yerr = result_dict[dataset]["chromosomal"][classifier]["yerr"]
			random_result_yerr = result_dict[dataset]["random"][classifier]["yerr"]
			# print(chromosomal_result, random_result)
			plt.bar([0 + bar_width*i, 1 + bar_width*i], [random_result_mean, chromosomal_result_mean], bar_width, alpha=alpha, yerr=[random_result_yerr, chromosomal_result_yerr],color=colorList[i],label=classifier)
		plt.ylabel('F-measure')
		plt.xticks([0 + bar_width*1, 1 + bar_width*1], ("random cv", "chromosomal cv"))
		# plt.title(f"pseudo ep2vec in {dataset} K562")
		plt.legend()
		plt.savefig(f"{dataset}_{args.cell_line}_F1.png",dpi=130,bbox_inches = 'tight', pad_inches = 0)
		plt.show()


def make_F1_barGraph(datasetNames, cell_line, classifiers):
	result_dict = get_F1_Dict(datasetNames, cell_line, classifiers)
	fig, ax = plt.subplots(3, 2, tight_layout=True, figsize=(9,8))
	bar_width = 0.10
	alpha = 0.8
	colorList = ["blue", "red", "green", "pink", "gold", "purple", "green"]
	for i, dataset in enumerate(datasetNames): # 調査するdatasetの数だけグラフを作成する．
		datasetName = dataset
		if dataset == "ep2vec":
			datasetName = "EP2vec"
		elif dataset == "new":
			datasetName = "maxflow based"
		for _ in range(2):
			ax[i, _].tick_params(bottom=False)
			ax[i, _].set_ylim([0, 1])
			ax[i, _].set_title(f"{datasetName} dataset on {cell_line}", loc="center")
		for j, classifier in enumerate(classifiers):
			if classifier[0] == "G":
				chromosomal_result_mean = result_dict[dataset]["chromosomal"][classifier]["mean"]
				random_result_mean = result_dict[dataset]["random"][classifier]["mean"]
				chromosomal_result_yerr = result_dict[dataset]["chromosomal"][classifier]["yerr"]
				random_result_yerr = result_dict[dataset]["random"][classifier]["yerr"]
				ax[i, 0].bar([0 + bar_width*j, 0.5 + bar_width*j], [random_result_mean, chromosomal_result_mean], bar_width, alpha=alpha, yerr=[random_result_yerr, chromosomal_result_yerr], error_kw={"capsize":5}, color=colorList[j],label=classifier)
				plt.sca(ax[i, 0])
				plt.xticks([0 + bar_width*1, 0.5 + bar_width*1], ("random cv", "chromosomal cv"))
				ax[i, 0].legend(loc="upper right", fontsize=7)
			else :
				chromosomal_result_mean = result_dict[dataset]["chromosomal"][classifier]["mean"]
				random_result_mean = result_dict[dataset]["random"][classifier]["mean"]
				chromosomal_result_yerr = result_dict[dataset]["chromosomal"][classifier]["yerr"]
				random_result_yerr = result_dict[dataset]["random"][classifier]["yerr"]
				ax[i, 1].bar([0 + bar_width*(j-3), 0.5 + bar_width*(j-3)], [random_result_mean, chromosomal_result_mean], bar_width, alpha=alpha, yerr=[random_result_yerr, chromosomal_result_yerr], error_kw={"capsize":5}, color=colorList[j],label=classifier)
				plt.sca(ax[i, 1])
				plt.xticks([0 + bar_width*1, 0.5 + bar_width*1], ("random cv", "chromosomal cv"))
				ax[i, 1].legend(loc="upper right", fontsize=7)

			ax[i, 0].set_ylabel('F-measure', fontsize=10, labelpad=10)
			ax[i, 1].set_ylabel('F-measure', fontsize=10, labelpad=10)

	# set the spacing between subplots
	plt.subplots_adjust(
					wspace=20.0, 
					hspace=5
					)

	plt.savefig(f"F1.png",dpi=300)
	plt.show()


def get_precision_recall(resultDir_path):
	files = os.listdir(resultDir_path)
	files_file = [f for f in files if os.path.isfile(os.path.join(resultDir_path, f))]
	y_true = []
	y_prob = []
	for i, result_file in enumerate(files_file):
		df = pd.read_csv(os.path.join(resultDir_path, result_file))
		y_true.append(df["y_test"].tolist())
		y_prob.append(df["y_pred"].tolist())
	y_true = [x for li in y_true for x in li]
	y_prob = [x for li in y_prob for x in li]
	precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

	return {"precision": precision, "recall": recall, "thresholds": thresholds}


def get_precision_recall_dict(datasetNames, classifiers):
	result_dict = {}
	for datasetName in datasetNames:
		result_dict[datasetName] = {"random": {}, "chromosomal": {}}
		for classifier in classifiers:
			resultDir = os.path.join(os.path.dirname(__file__), "ep2vec_result", datasetName, "chromosomal", classifier)
			result_dict[datasetName]["chromosomal"][classifier] = get_precision_recall(resultDir)
			resultDir = os.path.join(os.path.dirname(__file__), "ep2vec_result", datasetName, "random", classifier)
			result_dict[datasetName]["random"][classifier] = get_precision_recall(resultDir)
	return result_dict


def make_precision_recall_curve(datasetNames, classifiers):
	result_dict = get_precision_recall_dict(datasetNames, classifiers)
	for dataset in datasetNames: # 調査するdatasetの数だけグラフを作成する．
		# plt.figure(figsize=(6, 4))
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
		colorList = ["blue", "red", "green", "pink", "gold", "purple", "green"]
		for i, classifier in enumerate(classifiers):
			chromosomal_pre = result_dict[dataset]["chromosomal"][classifier]["precision"]
			random_pre = result_dict[dataset]["random"][classifier]["precision"]
			chromosomal_rec = result_dict[dataset]["chromosomal"][classifier]["recall"]
			random_rec = result_dict[dataset]["random"][classifier]["recall"]
			chromosomal_thre = result_dict[dataset]["chromosomal"][classifier]["thresholds"]
			random_thre = result_dict[dataset]["random"][classifier]["thresholds"]
			print(chromosomal_pre[-10:])
			print(chromosomal_rec[-10:])
			print(chromosomal_thre[-10:])

			ax1.plot(random_rec, random_pre, color=colorList[i], label=classifier)
			ax2.plot(chromosomal_rec, chromosomal_pre, color=colorList[i], label=classifier)
		ax1.set_title("random cv")
		ax2.set_title("chromosomal cv")
		ax1.set_xlabel("recall")
		ax1.set_ylabel("precision")
		ax2.set_xlabel("recall")
		# ax2.set_ylabel("precision")
		ax1.set_aspect('equal')
		ax2.set_aspect('equal')
		lines, labels = fig.axes[-1].get_legend_handles_labels()
		fig.legend(lines, labels, loc = 'upper center')
		# plt.legend(loc='upper center')
		plt.tight_layout()
		# fig.set_figheight(15)
		# fig.set_figheight(5)
		plt.savefig(f"{dataset}_{args.cell_line}_precision-recall.png",dpi=300,pad_inches=0.1)
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="TargetFinderの正例トレーニングデータから新たにトレーニングデータを作成する")
	parser.add_argument("--research_name", help="", default="TargetFinder")
	parser.add_argument("--cell_line", help="細胞株", default="K562")
	args = parser.parse_args()

	cell_line_list = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]
	research_list = ["ep2vec", "TargetFinder", "new"]
	# for researchName in research_list:
	# 	for cell_line in cell_line_list:
	# 		args.research_name = researchName
	# 		args.cell_line = cell_line
	# 		make_PosNeg_figure(args)

	for cell_line in cell_line_list:
		make_F1_barGraph(["TargetFinder", "ep2vec", "new"], cell_line, ["GBRT_100", "GBRT_1000", "GBRT_4000", "KNN_5", "KNN_10", "KNN_15"])
	# make_precision_recall_curve(["new", "TargetFinder", "ep2vec"], ["GBRT_100", "GBRT_1000", "GBRT_4000"])