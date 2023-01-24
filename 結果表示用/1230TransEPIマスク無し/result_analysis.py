from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

fig_root = os.path.join(os.path.dirname(__file__), "fig")

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
	cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	datatype_list = ["original", "maxflow_2500000"]
	if dataname == "BENGI":
		column_list = ["$D_{BG}^{-}$", "$D_{BG}^{NIMF,2.5M}$"]
	else:
		column_list = ["$D_{TF}^{-}$", "$D_{TF}^{NIMF,2.5M}$"]
		cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]
	for train_cell_type in cell_type_list:
		hist_list = [[0 for i in range(len(datatype_list))] for j in range(len(cell_type_list) + 1)]
		for i, test_cell_type in enumerate(cell_type_list):
			for j, datatype in enumerate(datatype_list):
				filenames = glob.glob(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_{train_cell_type}_noScheduler(lr=0.0001)_noMSE", f"{train_cell_type}-{test_cell_type}*.txt"))
				# if len(filenames) == 0:
				# 	print(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_GM12878_noScheduler(lr=0.0001)_noMSE_mask-wn", f"GM12878-{cell_type}*.txt"))
				# 	continue
				for file in filenames:
					df = pd.read_csv(file, sep="\t")
					true = df["true"].to_list()
					prob = df["pred"].to_list()

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


		for j in range(len(datatype_list)):
			avr = 0
			for i in range(len(cell_type_list)):
				avr += hist_list[i][j]
			avr /= len(cell_type_list)
			hist_list[-1][j] = avr

				
		df = pd.DataFrame(data=hist_list, index=cell_type_list + ["average"], columns=column_list)
		plt.figure(dpi=300)
		if metric == "MCC":
			sns.heatmap(df, annot=True, vmax=1, vmin=-1, center=0, cmap="Blues")
		elif metric != "MCC":
			sns.heatmap(df, annot=True, vmax=1, vmin=0, center=0, cmap="Blues")
		plt.tick_params(labelsize = 7.5)
		plt.xticks(fontsize=12)
		plt.title(f"{metric}")
		plt.savefig(os.path.join(fig_root, "heat", f"{metric},tool=TransEPI,dataset={dataname},train_cl={train_cell_type}"), transparent=True)
		# plt.close('all')


def make_bar(dataname, metric):
	cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	datatype_list = ["original", "maxflow_2500000"]
	if dataname == "BENGI":
		column_list = ["$D_{BG}^{-}$", "$N_{2.5M}({D_{BG}^{+}})$"]
		column_list = ["BG", "NIMF($d_{\max}=$2.5M)"]
	else:
		column_list = ["$D_{TF}^{-}$", "$N_{2.5M}({D_{TF}^{+}})$"]
		column_list = ["TF", "NIMF($d_{\max}=$2.5M)"]
		cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]

	for train_cell_type in cell_type_list:
		hist_list = [[0 for i in range(len(datatype_list))] for j in range(len(cell_type_list) + 1)]
		for i, test_cell_type in enumerate(cell_type_list):
			for j, datatype in enumerate(datatype_list):
				filenames = glob.glob(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_{train_cell_type}_noScheduler(lr=0.0001)_noMSE", f"{train_cell_type}-{test_cell_type}*.txt"))
				# if len(filenames) == 0:
				# 	print(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_GM12878_noScheduler(lr=0.0001)_noMSE_mask-wn", f"GM12878-{cell_type}*.txt"))
				# 	continue
				for file in filenames:
					df = pd.read_csv(file, sep="\t")
					true = df["true"].to_list()
					prob = df["pred"].to_list()

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
					elif metric == "recall":
						hist_list[i][j] = rec
					elif metric == "specificity":
						hist_list[i][j] = spc


		for j in range(len(datatype_list)):
			avr = 0
			for i in range(len(cell_type_list)):
				avr += hist_list[i][j]
			avr /= len(cell_type_list)
			hist_list[-1][j] = avr

		hist_list = np.array(hist_list).T.tolist()		
		plt.figure(dpi=300)
		width = 0.1
		X_axis = np.arange(len(cell_type_list + ["average"]))
		if len(column_list) % 2 == 0:
			left_x = X_axis - (width * (len(column_list) // 2)) + (width / 2)
		else:
			left_x = X_axis - (width * (len(column_list) // 2))

		for i, column in enumerate(column_list):
			plt.bar(left_x + width*i, hist_list[i], label = column, width=width)
			# for j, v in enumerate(hist_list[i]):
			# 	plt.text(left_x[j] + width*i, v * 0.9, f"{v:.2f}", color='black', va='center', ha="center", fontweight="bold", size=4)
		


		if metric == "MCC":
			plt.ylim((-0.2, 1.1))
		elif metric != "MCC":
			plt.ylim((0, 1.1))

		# plt.axhline(y=[0, 0.2, 0.4, 0.6, 0.8], xmin=-1, xmax=1, linestyle = "dotted")
		plt.axhline(0, 0, 1, color="black", linestyle="dotted", linewidth=1)
		plt.axhline(0.2, 0, 1, color="black", linestyle="dotted", linewidth=1)
		plt.axhline(0.4, 0, 1, color="black", linestyle="dotted", linewidth=1)
		plt.axhline(0.6, 0, 1, color="black", linestyle="dotted", linewidth=1)
		plt.axhline(0.8, 0, 1, color="black", linestyle="dotted", linewidth=1)	
		plt.axhline(1.0, 0, 1, color="black")
		showed_cell_type = ["HeLa-S3" if cell == "HeLa" else cell for cell in cell_type_list]
		plt.xticks(X_axis, showed_cell_type + ["average"])
		plt.ylabel(f"{metric}")
		plt.rc('legend', fontsize='x-small')
		plt.legend(ncol=len(column_list), loc="upper center")
		os.makedirs(os.path.dirname(os.path.join(fig_root, "bar", f"{metric}", f"{metric},tool=TransEPI,dataset={dataname},train_cl={train_cell_type}")), exist_ok="True")
		top_name = metric
		if metric == "balanced accuracy":
			top_name = "ba"
		plt.savefig(os.path.join(fig_root, "bar", f"{metric}", f"{top_name},tool=TransEPI,dataset={dataname},train_cl={train_cell_type}"))





def make_table(dataname):
	cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK", "HMEC"]
	datatype_list = ["original", "maxflow_2500000"]
	if dataname == "BENGI":
		column_list = ["$D_{BG}^{-}$", "$N_{2.5M}({D_{BG}^{+}})$"]
		column_list = ["BG", "NIMF($d_{max}=$2.5M)"]
	else:
		column_list = ["$D_{TF}^{-}$", "$N_{2.5M}({D_{TF}^{+}})$"]
		column_list = ["TF", "NIMF($d_{max}=$2.5M)"]
		cell_type_list = ["GM12878", "HeLa", "IMR90", "K562", "NHEK"]


	for i, datatype in enumerate(datatype_list):
		for train_cell_type in cell_type_list:
			table_list = [] # negative name, test cell, rec, spc, MCC, b-accuracy
			sum_of_metric = [0, 0, 0, 0]
			for j, test_cell_type in enumerate(cell_type_list):
				filenames = glob.glob(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_{train_cell_type}_noScheduler(lr=0.0001)_noMSE", f"{train_cell_type}-{test_cell_type}*.txt"))
				# if len(filenames) == 0:
				# 	print(os.path.join(os.path.dirname(__file__), "result", f"{dataname}", f"{dataname}_{datatype}_GM12878_noScheduler(lr=0.0001)_noMSE_mask-wn", f"GM12878-{cell_type}*.txt"))
				# 	continue
				for file in filenames:
					df = pd.read_csv(file, sep="\t")
					true = df["true"].to_list()
					prob = df["pred"].to_list()

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

					sum_of_metric[0] += rec
					sum_of_metric[1] += spc
					sum_of_metric[2] += mcc
					sum_of_metric[3] += ba
					table_list.append([column_list[i], test_cell_type, rec, spc, mcc, ba])
			table_list.append([column_list[i], "average", sum_of_metric[0]/len(cell_type_list), sum_of_metric[1]/len(cell_type_list), sum_of_metric[2]/len(cell_type_list), sum_of_metric[3]/len(cell_type_list)])
			table_df = pd.DataFrame(data=table_list, columns=["negative set", "test cell", "rec", "spc", "MCC", "b-accuracy"])
			table_df.to_csv(os.path.join(fig_root, "table", f"tool=TransEPI,dataset={dataname}_{datatype},train_cl={train_cell_type}.csv"))
					
					
					
					







		
INF = 9999999999

os.makedirs(fig_root, exist_ok=True)
for dataname in ["TargetFinder", "BENGI"]:
	for metric in ["recall", "specificity", "MCC", "balanced accuracy"]:
		# make_hist(dataname, metric)
		# make_heatmap(dataname, metric)
		make_bar(dataname, metric)
	# make_table(dataname)