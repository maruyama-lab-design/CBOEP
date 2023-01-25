import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
import argparse
import math

# f1, precision, recall, MCC
def get_f1(y_true, y_prob, threhold=0.5):
	y_pred = [1 if prob >= threhold else 0 for prob in y_prob]
	return f1_score(y_true, y_pred)

def get_precision(y_true, y_prob, threhold=0.5):
	y_pred = [1 if prob >= threhold else 0 for prob in y_prob]
	return precision_score(y_true, y_pred)

def get_recall(y_true, y_prob, threhold=0.5):
	y_pred = [1 if prob >= threhold else 0 for prob in y_prob]
	return recall_score(y_true, y_pred)

def get_mcc(y_true, y_prob, threhold=0.5):
	y_pred = [1 if prob >= threhold else 0 for prob in y_prob]
	return matthews_corrcoef(y_true, y_pred)

def get_balanced_accuracy(y_true, y_prob, threhold=0.5):
	y_pred = [1 if prob >= threhold else 0 for prob in y_prob]
	return balanced_accuracy_score(y_true, y_pred)


def get_averageScore_in_allFold(resultDir_path, metric):
	if not os.path.exists(resultDir_path):
			return 0
	files = os.listdir(resultDir_path)
	files_file = [f for f in files if os.path.isfile(os.path.join(resultDir_path, f))]
	# print(files_file)   # ['file1', 'file2.txt', 'file3.jpg']
	score = np.zeros(len(files_file))
	for i, result_file in enumerate(files_file):
		path = os.path.join(resultDir_path, result_file)
		if not os.path.exists(path):
			score[i] = 0
			continue
		df = pd.read_csv(path)
		y_true = df["y_test"].tolist()
		y_prob = df["y_pred"].tolist()

		if metric == "F1":
			score[i] = get_f1(y_true, y_prob)
		elif metric ==  "precision":
			score[i] = get_precision(y_true, y_prob)
		elif metric ==  "recall":
			score[i] = get_recall(y_true, y_prob)
		elif metric ==  "MCC":
			score[i] = get_mcc(y_true, y_prob)
		elif metric == "balanced_accuracy":
			score[i] = get_balanced_accuracy(y_true, y_prob)

	return np.mean(score)


def make_figure():
	metrics = ["F1", "precision", "recall", "MCC", "balanced_accuracy"]
	models = ["TargetFinder", "EP2vec", "PEP", "SPEID"]
	# models = ["TargetFinder", "EP2vec"]
	cell_line_list = ["GM12878", "K562", "HUVEC", "HeLa-S3", "NHEK", "IMR90"]
	datasets = ["TargetFinder", "EP2vec", "new"]

	for cell_line in cell_line_list:
		for metric in metrics:
			all_score = []
			for model in models:
				model_score =[]
				for dataset in datasets:

					path = os.path.join(os.path.dirname(__file__), "..", "result", model, cell_line, dataset)
					if not os.path.exists(path):
						pass
					score = get_averageScore_in_allFold(path, metric)
					model_score.append(score)

				all_score.append(model_score)

			all_score = np.array(all_score).T

			# 棒の配置位置、ラベルを用意
			x = np.array([1, 2, 3, 4])
			x_labels = models
						
			# マージンを設定
			margin = 0.2  #0 <margin< 1
			totoal_width = 0.8 - margin
			
			# 棒グラフをプロット
			fig = plt.figure()
			for i, h in enumerate(all_score):
				pos = x - totoal_width *( 1- (2*i+1)/len(all_score) )/2
				label = ""
				if i == 0:
					label = "TargetFinder dataset"
				elif i == 1:
					label = "EP2vec dataset"
				else:
					label = "maxflow based dataset"
				plt.bar(pos, h, width = totoal_width/len(all_score), label=label)
			
			# ラベルの設定
			plt.xticks(x, x_labels)

			plt.xlabel(f"model name")
			plt.title(f"{cell_line} {metric}")
			plt.ylim([0, 1])
			if metric == "MCC":
				plt.ylim([-1, 1])
			plt.legend()

			# plt.show()

			# os.system(f"mkdir -p /Users/ylwrvr/卒論/Koga_code/compare_model/make_figure/fig/{model}/{cell_line}/{dataset}")
			outputDir = f"/Users/ylwrvr/卒論/Koga_code/compare_model/make_figure/fig2/{cell_line}"
			os.system(f"mkdir -p {outputDir}")
			fig.savefig(f"{outputDir}/{metric}.png")


def make_confusion_matrix():
	models = ["TargetFinder", "EP2vec", "PEP", "SPEID"]
	cell_line_list = ["GM12878", "K562", "HUVEC", "HeLa-S3", "NHEK", "IMR90"]
	datasets = ["TargetFinder", "EP2vec", "new"]

	for cell_line in cell_line_list:
		for model in models:
			for dataset in datasets:
				path = os.path.join(os.path.dirname(__file__), "..", "result", model, cell_line, dataset)
				if not os.path.exists(path):
					continue
				files = os.listdir(path)
				files_file = [f for f in files if os.path.isfile(os.path.join(path, f))]
				y_true = []
				y_pred = []
				for i, result_file in enumerate(files_file):
					csv_path = os.path.join(path, result_file)
					if not os.path.exists(csv_path):
						continue
					df = pd.read_csv(csv_path)
					true = df["y_test"].tolist()
					prob = df["y_pred"].tolist()

					for (t, p) in zip(true, prob):
						y_true.append(t)
						if p >= 0.5:
							y_pred.append(1)
						else:
							y_pred.append(0)
				
				cm = confusion_matrix(y_true, y_pred)

				plt.figure(figsize=(12, 9))
				sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", annot_kws={"fontsize":30})
				plt.xlabel("predicted", fontsize=20)
				plt.ylabel("actual", fontsize=20)
				plt.tick_params(labelsize=18)
				os.system(f"mkdir -p /Users/ylwrvr/卒論/Koga_code/compare_model/make_figure/fig/{model}/{cell_line}/{dataset}")
				plt.savefig(f"/Users/ylwrvr/卒論/Koga_code/compare_model/make_figure/fig/{model}/{cell_line}/{dataset}/cm.png")

make_figure()
make_confusion_matrix()