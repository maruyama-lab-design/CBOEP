from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

fig_root = "D:\\ylwrv\\Koga_code\\結果表示用\\TransEPI(cl_wise)\\fig"

def pr_auc_score(y_true, y_prob):
	precision, recall, _ = mt.precision_recall_curve(y_true, y_prob)
	aupr = mt.auc(recall, precision)
	return aupr


def specifity_score(y_true, y_pred):
	tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

	# NPV
	# return tn / (tn + fn)

	# specificity
	#  TN / TN + FP
	return tn / (tn + fp)


def NPV_score(y_true, y_pred):
	tn, fp, fn, tp = mt.confusion_matrix(y_true, y_pred).ravel()

	# NPV
	return tn / (tn + fn)


def make_heatmap(dataname, metric):
	cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]
	datatype_list = ["org", "mf"]
	hist_list = [[0 for i in range(len(datatype_list))] for j in range(len(cell_type_list))]
	for i, cell_type in enumerate(cell_type_list):
		for j, datatype in enumerate(datatype_list):
			filenames = glob.glob(os.path.join(os.path.dirname(__file__), f"{dataname}_{datatype}", f"*{cell_type}_prediction*.txt"))
			# if len(filenames) == 0:
			# 	hist_list[i][j] = 
			for file in filenames:
				df = pd.read_table(file, skiprows=1, names=["y_test", "y_pred"])
				true = df["y_test"].to_list()
				prob = df["y_pred"].to_list()

				auc = mt.roc_auc_score(true, prob)
				aupr = pr_auc_score(true, prob)
				pred =  list(map(round, prob))
				true_prime = [int(i == 0) for i in true]
				pred_prime = [int(i == 0) for i in pred]
				f1 = mt.f1_score(true, pred)
				f1_prime = mt.f1_score(true_prime, pred_prime)
				ba = mt.balanced_accuracy_score(true, pred)
				pre = mt.precision_score(true, pred)
				rec = mt.recall_score(true, pred)
				spc = specifity_score(true, pred)
				npv = NPV_score(true, pred)
				mcc = mt.matthews_corrcoef(true, pred)
				
				if metric == "MCC":
					hist_list[i][j] = mcc
				elif metric == "balanced accuracy":
					hist_list[i][j] = ba
				elif metric == "AUC":
					hist_list[i][j] = auc
				elif metric == "AUPR":
					hist_list[i][j] = aupr


	df = pd.DataFrame(data=hist_list, index=cell_type_list, columns=["original", "maxflow_2500000"])
	plt.figure()
	if metric == "MCC":
		sns.heatmap(df, annot=True, vmax=1, vmin=-1, center=0)
	elif metric != "MCC":
		sns.heatmap(df, annot=True, vmax=1, vmin=0, center=0)
	plt.tick_params(labelsize = 7.5)
	plt.xticks(rotation=10)
	plt.title(f"{metric}")
	plt.savefig(f"{metric},tool=TargetFinder,dataset={dataname},train_cl=GM12878")
	# plt.close('all')




def make_hist(dataname, metric):
	cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]
	datatype_list = ["org", "mf"]
	thresholds = [0.05 * i for i in range(21)]
	hist_list = [[0 for i in range(len(datatype_list))] for j in range(len(cell_type_list))]

	colors = ["r", "b"]
	linestyles = ["solid", "dashed", "dashdot", "dotted"]
	fig = plt.figure(dpi=150)
	for i, cell_type in enumerate(cell_type_list):
		num = 230 + (i + 1)
		plt.subplot(num)
		for j, datatype in enumerate(datatype_list):
			filenames = glob.glob(os.path.join(os.path.dirname(__file__), f"{dataname}_{datatype}", f"*{cell_type}_prediction*.txt"))
			# if len(filenames) == 0:
			# 	hist_list[i][j] =
			df = None
			for file in filenames: # len = 1
				df = pd.read_table(file, skiprows=1, names=["y_test", "y_pred"])

			scores = []

			true = df["y_test"].to_list()
			prob = df["y_pred"].to_list()

			for threshold in thresholds:
				score = 0
				pred =  [1 if i >= threshold else 0 for i in prob]
				true_prime = [int(i == 0) for i in true]
				pred_prime = [int(i == 0) for i in pred]
				f1 = mt.f1_score(true, pred)
				f1_prime = mt.f1_score(true_prime, pred_prime)
				ba = mt.balanced_accuracy_score(true, pred)
				pre = mt.precision_score(true, pred)
				rec = mt.recall_score(true, pred)
				spc = specifity_score(true, pred)
				npv = NPV_score(true, pred)
				mcc = mt.matthews_corrcoef(true, pred)
				
				if metric == "MCC":
					score = mcc
					plt.ylim((-1, 1))
				elif metric == "balanced accuracy":
					score = ba
					plt.ylim((0, 1))

				scores.append(score)

			plt.plot(thresholds, scores, color=colors[j], linestyle=linestyles[j], linewidth=0.5, label=datatype)
			plt.tick_params(labelsize = 7.5)
			# plt.xticks(rotation=10)
			plt.title(cell_type, fontsize=10)
			plt.legend(loc=3, prop={'size': 6})

	# plt.xlabel("threshold")
	# plt.ylabel(f"{metric}")
	fig.supxlabel('threshold')
	fig.supylabel(f'{metric}')
	plt.tight_layout()
	plt.savefig(os.path.join(fig_root, "hist", f"{metric},tool=TransEPI,dataset={dataname},train_cl=GM12878"))




		
INF = 9999999999

for dataname in ["TF", "BG"]:
	for metric in ["MCC", "balanced accuracy"]:
		# make_heatmap(dataname, metric)
		make_hist(dataname, metric)