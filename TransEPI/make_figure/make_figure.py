import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import seaborn as sns

import collections



def NPV_score(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return tn / (tn + fn)

def specificity_score(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	return tn / (tn + fp)

def AUPR_score(y_true, y_prob):
	pre, rec, thresholds = precision_recall_curve(y_true, y_prob)
	aupr = auc(rec, pre)
	return aupr

def read_result(filename):
	df = pd.read_table(filename)
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	return y_true, y_pred, y_prob





def compare_org_vs_inf(cell="GM12878", train_pos="BG", test="TF"): # 4/14

	plt.figure()

	org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{train_pos}_org_{cell}","prediction", f"masked_{test}_org_{cell}.txt")
	y_true, y_pred, y_prob = read_result(org_file)

	AUC = roc_auc_score(y_true, y_prob)
	MCC = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	x = [1, 2, 3, 4, 5, 6, 7, 8]
	height = [AUC, MCC, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=-0.3, label="original")





	inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{train_pos}_INF_{cell}", "prediction", f"masked_{test}_org_{cell}.txt")
	y_true, y_pred, y_prob = read_result(inf_file)

	auc = roc_auc_score(y_true, y_prob)
	mcc = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	height = [auc, mcc, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")


	plt.xticks(x, ['AUC', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
	plt.ylim((-0.1, 1.2))
	plt.legend()
	plt.title(cell)
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"{cell},{train_pos}-{test}.jpg"))



def compare_org_vs_inf_cmn_test(train_cell="GM12878", test_cell="GM12878", dataname="BG"): # 4/14

	plt.figure()

	org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_org_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	y_true, y_pred, y_prob = read_result(org_file)

	AUC = roc_auc_score(y_true, y_prob)
	aupr = AUPR_score(y_true, y_prob)
	MCC = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	height = [AUC, aupr, MCC, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=-0.3, label=dataname)





	inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	y_true, y_pred, y_prob = read_result(inf_file)

	auc = roc_auc_score(y_true, y_prob)
	aupr = AUPR_score(y_true, y_prob)
	mcc = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	height = [auc, aupr, mcc, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")

	plt.grid(which = "major", axis = "y", color = "gray", alpha = 0.8, linestyle = "--", linewidth = 1)
	plt.xticks(x, ['AUC', 'AUPR', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
	plt.ylim((-0.1, 1.2))
	plt.legend()
	plt.title(f"{train_cell} -> {test_cell} (use common test)")
	os.makedirs(os.path.join(os.path.dirname(__file__), "bar"), exist_ok=True)
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"org_VS_inf_{train_cell}-{test_cell}_cmn.jpg"))

def fig5(train_cell="GM12878", test_cell_list=["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"], dataname="BG", metric="balanced accuracy"):

	table_data = []
	plt.figure()
	org_scores = []
	inf_scores = []
	org_sum = 0
	inf_sum = 0
	for cell in test_cell_list:
		org_file = os.path.join(os.path.dirname(__file__), "..", "holdout", "prediction", f"masked_{dataname}_org_{train_cell}",f"masked_{dataname}_cmn_{cell}.txt")
		org_y_true, org_y_pred, org_y_prob = read_result(org_file)
		inf_file = os.path.join(os.path.dirname(__file__), "..", "holdout", "prediction", f"masked_{dataname}_INF_{train_cell}",f"masked_{dataname}_cmn_{cell}.txt")
		inf_y_true, inf_y_pred, inf_y_prob = read_result(inf_file)
		org_score, inf_score = 0, 0
		if metric == "balanced accuracy":
			org_score = balanced_accuracy_score(org_y_true, org_y_pred)
			inf_score = balanced_accuracy_score(inf_y_true, inf_y_pred)
		elif metric == "recall":
			org_score = recall_score(org_y_true, org_y_pred)
			inf_score = recall_score(inf_y_true, inf_y_pred)
		elif metric == "specificity":
			org_score = specificity_score(org_y_true, org_y_pred)
			inf_score = specificity_score(inf_y_true, inf_y_pred)
		elif metric == "MCC":
			org_score = matthews_corrcoef(org_y_true, org_y_pred)
			inf_score = matthews_corrcoef(inf_y_true, inf_y_pred)
		elif metric == "AUC":
			org_score = roc_auc_score(org_y_true, org_y_prob)
			inf_score = roc_auc_score(inf_y_true, inf_y_prob)
		elif metric == "AUPR":
			org_score = AUPR_score(org_y_true, org_y_prob)
			inf_score = AUPR_score(inf_y_true, inf_y_prob)

		org_sum += org_score
		inf_sum += inf_score

		org_scores.append(org_score)
		inf_scores.append(inf_score)

	org_scores.append(org_sum / len(test_cell_list))
	inf_scores.append(inf_sum / len(test_cell_list))

	x = [1, 2, 3, 4, 5, 6, 7]

	print(f"___org {metric}___")
	print(org_scores)
	print(f"___inf {metric}___")
	print(inf_scores)
	print(f"__improvement from org to inf__")
	print(f"{(np.array(inf_scores)-np.array(org_scores)) / np.array(org_scores)}")

	plt.grid(axis="y", linestyle="dotted", color="black")
	plt.bar(x, org_scores, align="edge", width=-0.1, label=dataname, color="C0")
	plt.bar(x, inf_scores, align="edge", width=0.1, label="CBOEP($d_{max}=\infty$)", color="C4")

	table_data.append([dataname] + org_scores)
	table_data.append(["CBOEP($d_{max}=\infty$)"] + inf_scores)

	plt.xticks(x, test_cell_list + ["average"])
	plt.ylim((0, 1.15))
	plt.ylabel(f"{metric[0].upper() + metric[1:]}")
	plt.legend(loc="upper center", ncol=2)
	plt.axhline(y=1.0, color="black")
	os.makedirs(os.path.join(os.path.dirname(__file__), "bar_paper"), exist_ok=True)
	prefix = metric
	if metric == "balanced accuracy":
		prefix = "ba"
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar_paper", f"{prefix}_{dataname}_{train_cell}_cmn.png"), dpi=300)

	df = pd.DataFrame(data=table_data, columns=["name"] + test_cell_list + ["average"])
	os.makedirs(os.path.join(os.path.dirname(__file__), "csv_paper"), exist_ok=True)
	df.to_csv(os.path.join(os.path.dirname(__file__), "csv_paper", f"fig5_{metric}.csv"))


def fig4(train_cell="GM12878", test_cell_list=["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"], dataname="BG", metric="balanced accuracy"):

	if dataname == "TF":
		test_cell_list = ["GM12878", "HeLa-S3",  "IMR90", "K562", "NHEK"]

	table_data = []
	plt.figure()
	base_x = np.array(list(range(1, len(test_cell_list)+2)))
	bar_width = 0.1
	for i, datatype in enumerate(["org", "2500000", "5000000", "10000000", "INF"]):
		sum_score = 0
		scores = []
		for cell in test_cell_list:
			file = os.path.join(os.path.dirname(__file__), "..", "holdout", "prediction", f"masked_{dataname}_{datatype}_{train_cell}",f"masked_{dataname}_{datatype}_{cell}.txt")
			y_true, y_pred, y_prob = read_result(file)
			if metric == "balanced accuracy":
				score = balanced_accuracy_score(y_true, y_pred)
			elif metric == "recall":
				score = recall_score(y_true, y_pred)
			elif metric == "specificity":
				score = specificity_score(y_true, y_pred)
			elif metric == "MCC":
				score = matthews_corrcoef(y_true, y_pred)
			elif metric == "AUC":
				score = roc_auc_score(y_true, y_prob)
			elif metric == "AUPR":
				score = AUPR_score(y_true, y_prob)
			sum_score += score
			scores.append(score)
		scores.append(sum_score / len(test_cell_list))
		x = base_x + bar_width * (i-2)
		label = dataname
		if datatype == "2500000":
			label = "CBOEP($d_{max}=2.5$M)"
		elif datatype == "5000000":
			label = "CBOEP($d_{max}=5$M)"
		elif datatype == "10000000":
			label = "CBOEP($d_{max}=10$M)"
		elif datatype == "INF":
			label = "CBOEP($d_{max}=\infty$)"
		plt.bar(x, scores, align="center", width=bar_width, label=label)
		table_data.append([label] + scores)

	plt.grid(axis="y", linestyle="dotted", color="black")
	plt.xticks(base_x, test_cell_list + ["average"])
	plt.ylim((0, 1.15))
	plt.ylabel(f"{metric[0].upper() + metric[1:]}")
	plt.legend(loc="upper center", ncol=3, fontsize=7)
	plt.axhline(y=1.0, color="black")
	os.makedirs(os.path.join(os.path.dirname(__file__), "bar_paper"), exist_ok=True)
	prefix = metric
	if metric == "balanced accuracy":
		prefix = "ba"
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar_paper", f"{prefix}_{dataname}_{train_cell}_masked.png"), dpi=300)

	df = pd.DataFrame(data=table_data, columns=["name"]+test_cell_list+["average"])
	os.makedirs(os.path.join(os.path.dirname(__file__), "csv_paper"), exist_ok=True)
	df.to_csv(os.path.join(os.path.dirname(__file__), "csv_paper", f"fig4_{dataname}_{metric}.csv"))


def fig3(train_cell="GM12878", test_cell_list=["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"], dataname="BG", metric="balanced accuracy"):

	if dataname == "TF":
		test_cell_list = ["GM12878", "HeLa-S3",  "IMR90", "K562", "NHEK"]

	table_data = []
	plt.figure()
	base_x = np.array(list(range(1, len(test_cell_list)+2)))
	bar_width = 0.1
	for i, datatype in enumerate(["org", "2500000"]):
		sum_score = 0
		scores = []
		for cell in test_cell_list:
			file = os.path.join(os.path.dirname(__file__), "..", "holdout", "prediction", f"no_masked_{dataname}_{datatype}_{train_cell}",f"no_masked_{dataname}_{datatype}_{cell}.txt")
			y_true, y_pred, y_prob = read_result(file)
			if metric == "balanced accuracy":
				score = balanced_accuracy_score(y_true, y_pred)
			elif metric == "recall":
				score = recall_score(y_true, y_pred)
			elif metric == "specificity":
				score = specificity_score(y_true, y_pred)
			elif metric == "MCC":
				score = matthews_corrcoef(y_true, y_pred)
			elif metric == "AUC":
				score = roc_auc_score(y_true, y_prob)
			elif metric == "AUPR":
				score = AUPR_score(y_true, y_prob)
			sum_score += score
			scores.append(score)
		scores.append(sum_score / len(test_cell_list))
		x = base_x + bar_width * (i-2)
		label = dataname
		if datatype == "2500000":
			label = "CBOEP($d_{max}=2.5$M)"
		plt.bar(x, scores, align="center", width=bar_width, label=label)
		table_data.append([label] + scores)

	plt.grid(axis="y", linestyle="dotted", color="black")
	plt.xticks(base_x, test_cell_list + ["average"])
	plt.ylim((0, 1.15))
	plt.ylabel(f"{metric[0].upper() + metric[1:]}")
	plt.legend(loc="upper center", ncol=3, fontsize=7)
	plt.axhline(y=1.0, color="black")
	os.makedirs(os.path.join(os.path.dirname(__file__), "bar_paper"), exist_ok=True)
	prefix = metric
	if metric == "balanced accuracy":
		prefix = "ba"
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar_paper", f"{prefix}_{dataname}_{train_cell}_no_masked.png"), dpi=300)

	df = pd.DataFrame(data=table_data, columns=["name"]+test_cell_list+["average"])
	os.makedirs(os.path.join(os.path.dirname(__file__), "csv_paper"), exist_ok=True)
	df.to_csv(os.path.join(os.path.dirname(__file__), "csv_paper", f"fig3_{dataname}_{metric}.csv"))


def compare_org_vs_inf_cmn_test_kfold(train_cell="GM12878", test_cell="GM12878", dataname="BG", k=10): # 4/14

	plt.figure()

	scores = np.zeros((k, 9))
	for fold in range(k):
		# continue # !!
		org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_org_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
		y_true, y_pred, y_prob = read_result(org_file)

		scores[fold][0] = AUC = roc_auc_score(y_true, y_prob)
		scores[fold][1] = AUPR = AUPR_score(y_true, y_prob)
		scores[fold][2] = MCC = matthews_corrcoef(y_true, y_pred)
		scores[fold][3] = b_a = balanced_accuracy_score(y_true, y_pred)
		scores[fold][4] = f1 = f1_score(y_true, y_pred)
		scores[fold][5] = pre = precision_score(y_true, y_pred)
		scores[fold][6] = rec = recall_score(y_true, y_pred)
		scores[fold][7] = npv = NPV_score(y_true, y_pred)
		scores[fold][8] = spc = specificity_score(y_true, y_pred)

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y_mean = np.mean(scores, axis=0)
	y_std = np.std(scores, axis=0) / np.sqrt(k)
	print(y_mean)
	plt.bar(x, y_mean, yerr=y_std, capsize=0.2, align="edge", width=-0.3, label=dataname)



	scores = np.zeros((k, 9))
	for fold in range(k):
		inf_file = os.path.join(os.path.dirname(__file__), "..", "cross_val", f"fold{fold}", "prediction", f"masked_{dataname}_INF_{train_cell}", f"masked_{dataname}_cmn_{test_cell}.txt")
		y_true, y_pred, y_prob = read_result(inf_file)

		scores[fold][0] = AUC = roc_auc_score(y_true, y_prob)
		scores[fold][1] = AUPR = AUPR_score(y_true, y_prob)
		scores[fold][2] = MCC = matthews_corrcoef(y_true, y_pred)
		scores[fold][3] = b_a = balanced_accuracy_score(y_true, y_pred)
		scores[fold][4] = f1 = f1_score(y_true, y_pred)
		scores[fold][5] = pre = precision_score(y_true, y_pred)
		scores[fold][6] = rec = recall_score(y_true, y_pred)
		scores[fold][7] = npv = NPV_score(y_true, y_pred)
		scores[fold][8] = spc = specificity_score(y_true, y_pred)

	x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
	y_mean = np.mean(scores, axis=0)
	y_std = np.std(scores, axis=0) / np.sqrt(k)
	print(y_mean)
	plt.bar(x, y_mean, yerr=y_std, capsize=0.2, align="edge", width=0.3, label="NIMF(d_max=inf)")

	plt.grid(which = "major", axis = "y", color = "gray", alpha = 0.8, linestyle = "--", linewidth = 1)
	plt.xticks(x, ['AUC', 'AUPR', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
	plt.ylim((-0.1, 1.2))
	plt.legend()
	plt.title(f"{train_cell} -> {test_cell} (use common test)")
	os.makedirs(os.path.join(os.path.dirname(__file__), "bar_cross_val"), exist_ok=True)
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar_cross_val", f"org_VS_inf_{train_cell}-{test_cell}_cmn.jpg"))




def compare_2500000_vs_inf_cmn_test(train_cell="GM12878", test_cell="GM12878", dataname="BG"): # 4/14

	plt.figure()

	org_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_2500000_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	y_true, y_pred, y_prob = read_result(org_file)

	AUC = roc_auc_score(y_true, y_prob)
	MCC = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	x = [1, 2, 3, 4, 5, 6, 7, 8]
	height = [AUC, MCC, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=-0.3, label="NIMF(d_max=2.5M)")





	inf_file = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	y_true, y_pred, y_prob = read_result(inf_file)

	auc = roc_auc_score(y_true, y_prob)
	mcc = matthews_corrcoef(y_true, y_pred)
	b_a = balanced_accuracy_score(y_true, y_pred)
	f1 = f1_score(y_true, y_pred)
	pre = precision_score(y_true, y_pred)
	rec = recall_score(y_true, y_pred)
	npv = NPV_score(y_true, y_pred)
	spc = specificity_score(y_true, y_pred)

	height = [auc, mcc, b_a, f1, pre, rec, npv, spc] # y_org
	print(height)
	plt.bar(x, height, align="edge", width=0.3, label="NIMF(d_max=inf)")

	plt.grid(which = "major", axis = "y", color = "gray", alpha = 0.8, linestyle = "--", linewidth = 1)
	plt.xticks(x, ['AUC', 'MCC', 'b-acc', "F", "Pre", "Rec", "NPV", "Spec"])
	plt.ylim((-0.1, 1.2))
	plt.legend()
	plt.title(f"{train_cell} -> {test_cell} (use common test)")
	plt.savefig(os.path.join(os.path.dirname(__file__), "bar", f"2500000_VS_inf_{train_cell}-{test_cell}_cmn.jpg"))


def make_confusion_matrix_cmn_test(dataname="BG", datatype="org", train_cell="GM12878", test_cell="GM12878"):
	filename = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_{datatype}_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	y_true, y_pred, y_prob = read_result(filename)
	cm = confusion_matrix(y_true, y_pred)

	plt.figure()
	sns.heatmap(cm, square=True, cbar=True, annot=True, cmap='Blues', fmt=',d')
	plt.xlabel("Prediction", fontsize=13)
	plt.ylabel("Actual", fontsize=13)
	os.makedirs(os.path.join(os.path.dirname(__file__), "confusion_matrix"), exist_ok=True)
	plt.savefig(os.path.join(os.path.dirname(__file__), "confusion_matrix", f"{dataname}_{datatype}_{train_cell}-{test_cell}_cmn.jpg"))



def FP_ratio_by_distance_rank_cmn_test(dataname="BG", datatype="org", train_cell="GM12878", test_cell="GM12878", sorting="descending"):
	# org
	filename = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_org_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	df = pd.read_table(filename)
	if sorting == "ascending":
		df = df.sort_values('distance', ascending=True)
	else:
		df = df.sort_values('distance', ascending=False)
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	distance = df["distance"].to_list()
	
	is_fp = [0] * len(y_true)
	for i in range(len(y_true)):
		if y_true[i] == 0 and y_pred[i] == 1:
			is_fp[i] = 1
	
	for i in range(1, len(is_fp)):
		is_fp[i] += is_fp[i-1]

	if is_fp[-1] != 0:
		fp_ratio_org = list(map(lambda x: x/is_fp[-1], is_fp))
	else:
		fp_ratio_org = is_fp

	# INF
	filename = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	df = pd.read_table(filename)
	if sorting == "ascending":
		df = df.sort_values('distance', ascending=True)
	else:
		df = df.sort_values('distance', ascending=False)
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	distance = df["distance"].to_list()
	
	is_fp = [0] * len(y_true)
	for i in range(len(y_true)):
		if y_true[i] == 0 and y_pred[i] == 1:
			is_fp[i] = 1
	
	for i in range(1, len(is_fp)):
		is_fp[i] += is_fp[i-1]

	if is_fp[-1] != 0:
		fp_ratio_INF = list(map(lambda x: x/is_fp[-1], is_fp))
	else:
		fp_ratio_INF = is_fp


	plt.figure()
	X = list(range(1, len(fp_ratio_org)+1))
	plt.plot(X, fp_ratio_org, label="BENGI")
	plt.plot(X, fp_ratio_INF, label="NIMF(max_d=INF)")
	plt.xlabel(f"distance rank ({sorting})", fontsize=13)
	plt.ylabel("FP ratio", fontsize=13)
	plt.ylim((-0.1, 1.1))
	plt.grid(True)
	plt.legend()
	plt.title(f"{train_cell}->{test_cell}")
	# plt.show()
	os.makedirs(os.path.join(os.path.dirname(__file__), "FP_raio"), exist_ok=True)
	plt.savefig(os.path.join(os.path.dirname(__file__), "FP_raio", f"{dataname}_{train_cell}-{test_cell}_cmn.jpg"))




def FN_ratio_by_distance_rank_cmn_test(dataname="BG", datatype="org", train_cell="GM12878", test_cell="GM12878", sorting="descending"):
	# org
	filename = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_org_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	df = pd.read_table(filename)
	if sorting == "ascending":
		df = df.sort_values('distance', ascending=True)
	else:
		df = df.sort_values('distance', ascending=False)
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	distance = df["distance"].to_list()
	
	is_fn = [0] * len(y_true)
	for i in range(len(y_true)):
		if y_true[i] == 1 and y_pred[i] == 0:
			is_fn[i] = 1
	
	for i in range(1, len(is_fn)):
		is_fn[i] += is_fn[i-1]

	if is_fn[-1] != 0:
		fn_ratio_org = list(map(lambda x: x/is_fn[-1], is_fn))
	else:
		fn_ratio_org = is_fn

	# INF
	filename = os.path.join(os.path.dirname(__file__), "..", "out", "model", f"masked_{dataname}_INF_{train_cell}","prediction", f"masked_{dataname}_cmn_{test_cell}.txt")
	df = pd.read_table(filename)
	if sorting == "ascending":
		df = df.sort_values('distance', ascending=True)
	else:
		df = df.sort_values('distance', ascending=False)
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	distance = df["distance"].to_list()
	
	is_fn = [0] * len(y_true)
	for i in range(len(y_true)):
		if y_true[i] == 1 and y_pred[i] == 0:
			is_fn[i] = 1
	
	for i in range(1, len(is_fn)):
		is_fn[i] += is_fn[i-1]

	if is_fn[-1] != 0:
		fn_ratio_INF = list(map(lambda x: x/is_fn[-1], is_fn))
	else:
		fn_ratio_INF = is_fn


	plt.figure()
	X = list(range(1, len(fn_ratio_org)+1))
	plt.plot(X, fn_ratio_org, label="BENGI")
	plt.plot(X, fn_ratio_INF, label="NIMF(max_d=INF)")
	plt.xlabel(f"distance rank ({sorting})", fontsize=13)
	plt.ylabel("FP ratio", fontsize=13)
	plt.ylim((-0.1, 1.1))
	plt.grid(True)
	plt.title(f"{train_cell}->{test_cell}")
	plt.legend()
	# plt.show()
	os.makedirs(os.path.join(os.path.dirname(__file__), "FN_raio"), exist_ok=True)
	plt.savefig(os.path.join(os.path.dirname(__file__), "FN_raio", f"{dataname}_{train_cell}-{test_cell}_cmn.jpg"))


def TnFp_freq_by_distance_on_cmn_test(dataname="BG", datatype="org", train_cell="GM12878", test_cell="GM12878", bottom_p=100, bin_size=10000):
	# org
	filename = os.path.join(os.path.dirname(__file__), "..", "holdout", "prediction", f"masked_{dataname}_{datatype}_{train_cell}", f"masked_{dataname}_cmn_{test_cell}.txt")
	df = pd.read_table(filename)
	df = df.sort_values(by="distance").reset_index()
	df = df.loc[:int(len(df) * (bottom_p / 100)), :]

	
	y_true = df["true"].to_list()
	y_prob = df["pred"].to_list()
	y_pred = list(map(round, y_prob))
	dist = df["distance"].to_list()

	tp_dist = []
	tn_dist = []
	fp_dist = []
	fn_dist = []
	
	for i in range(len(y_true)):
		if y_true[i] == 0 and y_pred[i] == 0:
			tn_dist.append(dist[i])
		elif y_true[i] == 0 and y_pred[i] == 1:
			fp_dist.append(dist[i])
		elif y_true[i] == 1 and y_pred[i] == 1:
			tp_dist.append(dist[i])
		elif y_true[i] == 1 and y_pred[i] == 0:
			fn_dist.append(dist[i])

	tn_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), tn_dist))
	fp_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), fp_dist))
	tp_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), tp_dist))
	fn_dist = list(map(lambda x: ((x-1)//bin_size)*bin_size+(bin_size/2), fn_dist))

			
	if len(tn_dist) == 0:
		tn_X_max = 0
	else:
		tn_X_max = max(tn_dist)
	
	if len(fp_dist) == 0:
		fp_X_max = 0
	else:
		fp_X_max = max(fp_dist)

	if len(tp_dist) == 0:
		tp_X_max = 0
	else:
		tp_X_max = max(tp_dist)
	
	if len(fn_dist) == 0:
		fn_X_max = 0
	else:
		fn_X_max = max(fn_dist)	

	tn_c = collections.Counter(tn_dist)
	fp_c = collections.Counter(fp_dist)
	tp_c = collections.Counter(tp_dist)
	fn_c = collections.Counter(fn_dist)


	tn_X = []
	tn_Y = []
	fp_X = []
	fp_Y = []
	for n_bin in range(max(int(tn_X_max*10)//(bin_size*10)+1, int(fp_X_max*10)//(bin_size*10)+1)):
		x = n_bin * bin_size + (bin_size / 2)
		tn_X.append(x)
		tn_Y.append(tn_c[x])
		fp_X.append(x)
		fp_Y.append(fp_c[x])
		# print(tn_X, tn_Y)

	outdir = os.path.join(os.path.dirname(__file__), "distance_distribution_of_all_neg")
	os.makedirs(outdir, exist_ok=True)


	tn_X, tn_Y = np.array(tn_X), np.array(tn_Y)
	fp_X, fp_Y = np.array(fp_X), np.array(fp_Y)
	plt.figure()
	plt.bar(tn_X, tn_Y, width=bin_size, align="center", color="green", alpha=0.6, label="True Negative")
	plt.bar(fp_X, fp_Y, width=bin_size, align="center", color="purple", alpha=0.6, label="False Positive", bottom=tn_Y)
	plt.xlabel(f"EP distance (bin size {bin_size})")
	plt.ylabel(f"freq")
	if datatype == "org":
		title = f"train {dataname} {datatype} {train_cell} model, test on {test_cell} all neg pairs"
	else:
		title = f"train CBOEP {datatype} {train_cell} model, test on {test_cell} all neg pairs"	
	plt.title(title)
	plt.legend()
	plt.savefig(os.path.join(outdir, f"{dataname}_{datatype}_{train_cell}-{test_cell}_bin{bin_size}_bottom{bottom_p}.png"))



	tp_X = []
	tp_Y = []
	fn_X = []
	fn_Y = []
	for n_bin in range(max(int(tp_X_max*10)//(bin_size*10)+1, int(fn_X_max*10)//(bin_size*10)+1)):
		x = n_bin * bin_size + (bin_size / 2)
		tp_X.append(x)
		tp_Y.append(tp_c[x])
		fn_X.append(x)
		fn_Y.append(fn_c[x])
	outdir = os.path.join(os.path.dirname(__file__), "distance_distribution_of_all_pos")
	os.makedirs(outdir, exist_ok=True)


	tp_X, tp_Y = np.array(tp_X), np.array(tp_Y)
	fn_X, fn_Y = np.array(fn_X), np.array(fn_Y)
	plt.figure()
	plt.bar(tp_X, tp_Y, width=bin_size, align="center", color="green", alpha=0.6, label="True Positive")
	plt.bar(fn_X, fn_Y, width=bin_size, align="center", color="purple", alpha=0.6, label="False Negative", bottom=tp_Y)
	plt.xlabel(f"EP distance (bin size {bin_size})")
	plt.ylabel(f"freq")
	if datatype == "org":
		title = f"train {dataname} {datatype} {train_cell} model, test on {test_cell} pos pairs"
	else:
		title = f"train CBOEP {datatype} {train_cell} model, test on {test_cell} pos pairs"	

	plt.title(title)
	plt.legend()
	plt.savefig(os.path.join(outdir, f"{dataname}_{datatype}_{train_cell}-{test_cell}_bin{bin_size}_bottom{bottom_p}.png"))



for datatype in ["INF", "org"]:
	for train_cell in ["GM12878"]:
		for test_cell in ["GM12878", "HeLa-S3", "NHEK", "IMR90", "HMEC", "K562"]:
			# FP_ratio_by_distance_rank_cmn_test(datatype=datatype, train_cell=train_cell, test_cell=test_cell)
			# FN_ratio_by_distance_rank_cmn_test(datatype=datatype, train_cell=train_cell, test_cell=test_cell)
			TnFp_freq_by_distance_on_cmn_test(datatype=datatype, train_cell=train_cell, test_cell=test_cell, bottom_p=25, bin_size=1000)
		

exit()


# for datatype in ["INF", "org"]:
# 	for train_cell in ["GM12878", "HeLa-S3", "NHEK", "IMR90", "HMEC", "K562"]:
# 		for test_cell in ["GM12878", "HeLa-S3", "NHEK", "IMR90", "HMEC", "K562"]:
# 			make_confusion_matrix_cmn_test(datatype=datatype, train_cell=train_cell, test_cell=test_cell)

for metric in ["balanced accuracy", "recall", "specificity"]:
	# compare_org_vs_inf_cmn_test_by_cell(metric=metric)
	fig3(metric=metric, dataname="BG")
	fig3(metric=metric, dataname="TF")
	fig4(metric=metric, dataname="BG")
	fig4(metric=metric, dataname="TF")
	fig5(metric=metric)
exit()


for train_cell in ["GM12878", "HeLa-S3", "NHEK", "IMR90", "HMEC", "K562"]:
	for test_cell in ["GM12878", "HeLa-S3", "NHEK", "IMR90", "HMEC", "K562"]:
		# compare_2500000_vs_inf_cmn_test(train_cell="GM12878", test_cell=cell, dataname="BG")
		compare_org_vs_inf_cmn_test(train_cell=train_cell, test_cell=test_cell, dataname="BG")
		# compare_org_vs_inf_cmn_test_kfold(train_cell=train_cell, test_cell=test_cell, dataname="BG", k=10)