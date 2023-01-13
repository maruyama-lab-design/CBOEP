from operator import index
import pandas as pd
import os
import glob
import sklearn
import sklearn.metrics as mt

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


def result_analysis(dataname, datatype):
	column_names = ["cell line", "tree", "depth", "learning rate", "AUC", "AUPR", "F", "F\'", "balanced accuracy", "precison", "recall", "specificity", "NPV", "MCC"]
	data_list = []
	for filename in glob.glob(f"../{region}/{dataname}_{datatype}/*.csv"):
		rn = os.path.basename(filename).split(".")[0]
		cl, tree, depth, alpha = rn.split(",")
		df = pd.read_csv(filename)
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
		
		data_list.append([cl, tree, depth, alpha, auc, aupr, f1, f1_prime, ba, pre, rec, spc, npv, mcc])

	csv_df = pd.DataFrame(data=data_list, index=None, columns=column_names)
	csv_df.to_csv(f"../{region}/{dataname}_{datatype}_result.csv", sep="\t", index=False)
	# with pd.ExcelWriter(os.path.join(os.path.dirname(__file__), "result_analysis.xlsx"), mode="a") as writer:
	#     csv_df.to_excel(writer, sheet_name=f"TargetFinder")

		
INF = 9999999999

region = "epw"
for dataname in ["BENGI"]:
	for datatype in ["original", "maxflow_2500000", "maxflow_5000000", "maxflow_10000000", f"maxflow_{INF}"]:
		result_analysis(dataname, datatype)